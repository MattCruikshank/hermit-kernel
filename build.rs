use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

use anyhow::{Context, Result};
use llvm_tools::LlvmTools;

fn main() -> Result<()> {
	built::write_built_file().unwrap();
	metal_source_fingerprint()?;

	if env::var("CARGO_CFG_TARGET_ARCH").unwrap() == "x86_64"
		&& env::var_os("CARGO_FEATURE_SMP").is_some()
	{
		assemble_x86_64_smp_boot()?;
	}

	Ok(())
}

fn assemble_x86_64_smp_boot() -> Result<()> {
	let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());

	let boot_s = Path::new("src/arch/x86_64/kernel/boot.s");
	let boot_ll = out_dir.join("boot.ll");
	let boot_bc = out_dir.join("boot.bc");
	let boot_bin = out_dir.join("boot.bin");

	let llvm_as = binutil("llvm-as");
	let lld = lld();

	let assembly = fs::read_to_string(boot_s)?;

	let mut llvm_file = File::create(&boot_ll)?;
	writeln!(
		&mut llvm_file,
		r#"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-none-elf"

module asm "
{assembly}
"
"#
	)?;
	llvm_file.flush()?;
	drop(llvm_file);

	let status = Command::new(&llvm_as)
		.arg("-o")
		.arg(&boot_bc)
		.arg(boot_ll)
		.status()
		.with_context(|| format!("Failed to run llvm-as from {}", llvm_as.display()))?;
	assert!(status.success());

	let status = Command::new(&lld)
		.arg("-flavor")
		.arg("gnu")
		.arg("--image-base=0x8000")
		.arg("--section-start=.text=0x8000")
		.arg("--oformat=binary")
		.arg("-o")
		.arg(&boot_bin)
		.arg(&boot_bc)
		.status()
		.with_context(|| format!("Failed to run lld from {}", lld.display()))?;
	assert!(status.success());

	println!("cargo:rerun-if-changed={}", boot_s.display());
	Ok(())
}

fn lld() -> PathBuf {
	let rust_lld = binutil("rust-lld");

	if rust_lld.exists() {
		return rust_lld;
	}

	binutil("lld")
}

fn binutil(name: &str) -> PathBuf {
	let exe = format!("{name}{}", env::consts::EXE_SUFFIX);

	if let Some(tool) = LlvmTools::new()
		.ok()
		.and_then(|llvm_tools| llvm_tools.tool(&exe))
	{
		return tool;
	}

	PathBuf::from(exe)
}

/// METAL: a fingerprint of the kernel sources (`Cargo.toml`, `Cargo.lock`, everything under
/// `src/`), printed at boot as `METAL kernel source: <hex>`. METAL's `xtask` computes the same
/// value from the source it meant to build and fails if the booted kernel reports another one —
/// so a stale kernel build can't pass for a fresh one. FNV-1a 64 over each file's relative path
/// (with `/`), a 0 byte, its contents and a 0 byte, in sorted path order. Keep in sync with
/// `kernel_source_hash` in METAL's `xtask/src/main.rs`.
fn metal_source_fingerprint() -> Result<()> {
	fn walk(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
		for entry in fs::read_dir(dir)? {
			let path = entry?.path();
			if path.is_dir() {
				walk(&path, files)?;
			} else {
				files.push(path);
			}
		}
		Ok(())
	}
	let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
	let mut files = vec![root.join("Cargo.toml"), root.join("Cargo.lock")];
	walk(&root.join("src"), &mut files)?;
	let mut entries: Vec<(String, PathBuf)> = files
		.into_iter()
		.filter(|f| f.exists())
		.map(|f| {
			let rel = f.strip_prefix(&root).unwrap().to_string_lossy().replace('\\', "/");
			(rel, f)
		})
		.collect();
	entries.sort();
	let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
	let mut feed = |bytes: &[u8]| {
		for b in bytes {
			hash ^= u64::from(*b);
			hash = hash.wrapping_mul(0x0100_0000_01b3);
		}
	};
	for (rel, path) in &entries {
		feed(rel.as_bytes());
		feed(&[0]);
		feed(&fs::read(path)?);
		feed(&[0]);
	}
	println!("cargo:rustc-env=METAL_KERNEL_SOURCE={hash:016x}");
	println!("cargo:rerun-if-changed=src");
	println!("cargo:rerun-if-changed=Cargo.toml");
	println!("cargo:rerun-if-changed=Cargo.lock");
	Ok(())
}
