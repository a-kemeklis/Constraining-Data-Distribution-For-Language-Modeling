/*!
bin/main.rs — Command-line interface for the whitelist tokenizer.

Usage
-----
    # Encode a file
    wl_tok encode --tokenizer tokenizer/ --input text.txt --output ids.bin

    # Decode a binary ids file
    wl_tok decode --tokenizer tokenizer/ --input ids.bin

    # Run sanity checks
    wl_tok check --tokenizer tokenizer/

    # Benchmark throughput (reads stdin or a file)
    wl_tok bench --tokenizer tokenizer/ --input text.txt
*/

use std::fs;
use std::io::{self, BufRead, Read, Write};
use std::path::Path;
use std::time::Instant;

use whitelist_tokenizer_rs::WhitelistTokenizer;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: wl_tok <command> [options]");
        eprintln!("Commands: encode, decode, check, bench");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "encode" => cmd_encode(&args[2..])?,
        "decode" => cmd_decode(&args[2..])?,
        "check"  => cmd_check(&args[2..])?,
        "bench"  => cmd_bench(&args[2..])?,
        other    => {
            eprintln!("Unknown command: {other}");
            std::process::exit(1);
        }
    }
    Ok(())
}

fn find_flag<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    for i in 0..args.len().saturating_sub(1) {
        if args[i] == flag {
            return Some(&args[i + 1]);
        }
    }
    None
}

fn load_tok(args: &[String]) -> anyhow::Result<WhitelistTokenizer> {
    let dir = find_flag(args, "--tokenizer")
        .ok_or_else(|| anyhow::anyhow!("--tokenizer <dir> required"))?;
    WhitelistTokenizer::from_directory(Path::new(dir))
}

// ---------------------------------------------------------------------------
// encode: read lines from --input (or stdin), write binary u32 LE ids
// ---------------------------------------------------------------------------

fn cmd_encode(args: &[String]) -> anyhow::Result<()> {
    let tok = load_tok(args)?;
    let prepend_bos = args.contains(&"--bos".to_string());
    let input_path = find_flag(args, "--input");
    let output_path = find_flag(args, "--output");

    let lines: Box<dyn Iterator<Item = String>> = match input_path {
        Some(p) => {
            let f = fs::File::open(p)?;
            Box::new(io::BufReader::new(f).lines().map(|l| l.unwrap()))
        }
        None => Box::new(io::BufReader::new(io::stdin()).lines().map(|l| l.unwrap())),
    };

    let mut out: Box<dyn Write> = match output_path {
        Some(p) => Box::new(io::BufWriter::new(fs::File::create(p)?)),
        None => Box::new(io::BufWriter::new(io::stdout())),
    };

    for line in lines {
        let ids = tok.encode(&line, prepend_bos);
        // Write: [4-byte LE count][ids as u32 LE]
        let count = ids.len() as u32;
        out.write_all(&count.to_le_bytes())?;
        for id in ids {
            out.write_all(&id.to_le_bytes())?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// decode: read binary u32 LE ids, print decoded text
// ---------------------------------------------------------------------------

fn cmd_decode(args: &[String]) -> anyhow::Result<()> {
    let tok = load_tok(args)?;
    let input_path = find_flag(args, "--input");

    let data = match input_path {
        Some(p) => fs::read(p)?,
        None => {
            let mut buf = Vec::new();
            io::stdin().read_to_end(&mut buf)?;
            buf
        }
    };

    let mut cursor = io::Cursor::new(&data);
    loop {
        let mut hdr = [0u8; 4];
        if cursor.read_exact(&mut hdr).is_err() {
            break;
        }
        let count = u32::from_le_bytes(hdr) as usize;
        let mut ids = vec![0u32; count];
        for id in ids.iter_mut() {
            let mut buf = [0u8; 4];
            cursor.read_exact(&mut buf)?;
            *id = u32::from_le_bytes(buf);
        }
        println!("{}", tok.decode(&ids));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// check: sanity-check the tokenizer against known test cases
// ---------------------------------------------------------------------------

fn cmd_check(args: &[String]) -> anyhow::Result<()> {
    let tok = load_tok(args)?;
    println!("WhitelistTokenizer loaded. vocab_size={}", tok.vocab_size());
    println!("  bos_id={} mask_id={} unk_id={} space_id={}", tok.bos_id(), tok.mask_id(), tok.unk_id(), tok.space_id());

    let cases: &[(&str, &str)] = &[
        ("The quick brown fox",       "plain words"),
        ("dog, cat.",                 "trailing punct"),
        ("dog's toy",                 "possessive"),
        ("don't won't can't",         "whole contractions"),
        ("well-known two-thirds",     "hyphenated compounds"),
        ("Hello wörld 42 🎉",         "non-ASCII / digit / emoji → UNK"),
        ("He said \"hello\" there.",  "double quotes"),
        ("It\u{2019}s a test.",       "curly apostrophe"),
        ("dog  ,  cat",               "multiple spaces around punct"),
        ("dog , cat",                 "space-padded comma"),
        ("dog,cat",                   "no-space comma"),
    ];

    let unk_id = tok.unk_id();
    for (input, desc) in cases {
        let ids = tok.encode(input, false);
        let n_unk = ids.iter().filter(|&&id| id == unk_id).count();
        let decoded = tok.decode(&ids);
        println!("\n  [{desc}]");
        println!("    input   : {input:?}");
        println!("    tokens  : {ids:?}");
        println!("    n_unk   : {n_unk}");
        println!("    decoded : {decoded:?}");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// bench: measure throughput
// ---------------------------------------------------------------------------

fn cmd_bench(args: &[String]) -> anyhow::Result<()> {
    let tok = load_tok(args)?;
    let input_path = find_flag(args, "--input")
        .ok_or_else(|| anyhow::anyhow!("--input <file> required for bench"))?;

    let text = fs::read_to_string(input_path)?;
    let lines: Vec<&str> = text.lines().collect();
    let total_chars: usize = lines.iter().map(|l| l.len()).sum();

    println!("Benchmarking on {} lines ({} chars) ...", lines.len(), total_chars);

    let t0 = Instant::now();
    let mut total_tokens = 0usize;
    for line in &lines {
        let ids = tok.encode(line, false);
        total_tokens += ids.len();
    }
    let elapsed = t0.elapsed();

    println!(
        "  {} tokens in {:.2}s  =  {:.2}M tok/s  ({:.2}M chars/s)",
        total_tokens,
        elapsed.as_secs_f64(),
        total_tokens as f64 / elapsed.as_secs_f64() / 1e6,
        total_chars as f64 / elapsed.as_secs_f64() / 1e6,
    );
    Ok(())
}
