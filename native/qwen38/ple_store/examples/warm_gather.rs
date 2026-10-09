use std::ffi::{c_char, CStr, CString};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use q38_ple_store::{q38_ple_close, q38_ple_last_error, q38_ple_open, q38_ple_read, q38_ple_stats};

const ROWS: usize = 131_072;
const ROW_BYTES: usize = 160;
const REPS: usize = 9;
const STATS_LEN: usize = 8;

struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn last_error() -> String {
    unsafe {
        CStr::from_ptr(q38_ple_last_error())
            .to_string_lossy()
            .into_owned()
    }
}

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn expected_rows(payload: &[u8], row_ids: &[i64]) -> Vec<u8> {
    let mut expected = Vec::with_capacity(row_ids.len() * ROW_BYTES);
    for &row_id in row_ids {
        let start = row_id as usize * ROW_BYTES;
        expected.extend_from_slice(&payload[start..start + ROW_BYTES]);
    }
    expected
}

fn run_case(name: &str, path: &Path, payload: &[u8], prewarm_ids: &[i64], patterns: &[Vec<i64>]) {
    let path = CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
    let paths: [*const c_char; 1] = [path.as_ptr()];
    let offsets = [0_u64];
    let row_counts = [ROWS as u64];
    let handle = unsafe {
        q38_ple_open(
            paths.as_ptr(),
            offsets.as_ptr(),
            row_counts.as_ptr(),
            1,
            ROW_BYTES,
            payload.len(),
            32,
            4096,
        )
    };
    assert!(!handle.is_null(), "q38_ple_open failed: {}", last_error());

    let mut prewarm_output = vec![0_u8; prewarm_ids.len() * ROW_BYTES];
    let cold_start = Instant::now();
    let status = unsafe {
        q38_ple_read(
            handle,
            prewarm_ids.as_ptr(),
            prewarm_ids.len(),
            prewarm_output.as_mut_ptr(),
            prewarm_output.len(),
        )
    };
    let cold = cold_start.elapsed();
    assert_eq!(status, 0, "q38_ple_read failed: {}", last_error());
    assert_eq!(prewarm_output, payload);

    let expected: Vec<Vec<u8>> = patterns
        .iter()
        .map(|row_ids| expected_rows(payload, row_ids))
        .collect();
    let max_output_len = patterns
        .iter()
        .map(|row_ids| row_ids.len() * ROW_BYTES)
        .max()
        .unwrap();
    let mut output = vec![0_u8; max_output_len];

    let mut before = [0_u64; STATS_LEN];
    let status = unsafe { q38_ple_stats(handle, before.as_mut_ptr(), before.len()) };
    assert_eq!(status, 0, "q38_ple_stats failed: {}", last_error());

    let mut samples = Vec::with_capacity(REPS);
    let mut timed_rows = 0_u64;
    for iteration in 0..REPS {
        let pattern_index = iteration % patterns.len();
        let row_ids = &patterns[pattern_index];
        let output_len = row_ids.len() * ROW_BYTES;
        let start = Instant::now();
        let status = unsafe {
            q38_ple_read(
                handle,
                row_ids.as_ptr(),
                row_ids.len(),
                output.as_mut_ptr(),
                output_len,
            )
        };
        samples.push(start.elapsed());
        assert_eq!(status, 0, "q38_ple_read failed: {}", last_error());
        assert_eq!(&output[..output_len], expected[pattern_index]);
        timed_rows += row_ids.len() as u64;
    }

    let mut after = [0_u64; STATS_LEN];
    let status = unsafe { q38_ple_stats(handle, after.as_mut_ptr(), after.len()) };
    assert_eq!(status, 0, "q38_ple_stats failed: {}", last_error());
    assert_eq!(after[0] - before[0], REPS as u64);
    assert_eq!(after[1] - before[1], timed_rows);
    assert_eq!(after[2] - before[2], timed_rows);
    assert_eq!(after[3] - before[3], 0);
    assert_eq!(after[4] - before[4], 0);

    let case_median = median(&mut samples);
    let samples_ms: Vec<f64> = samples
        .iter()
        .map(|sample| sample.as_secs_f64() * 1e3)
        .collect();
    println!(
        "case={name} pattern_count={} rows_per_pattern={:?}",
        patterns.len(),
        patterns.iter().map(Vec::len).collect::<Vec<_>>()
    );
    println!("case={name} cold_ms={:.3}", cold.as_secs_f64() * 1e3);
    println!("case={name} warm_samples_ms={samples_ms:?}");
    println!(
        "case={name} warm_median_ms={:.3}",
        case_median.as_secs_f64() * 1e3
    );
    println!("case={name} stats_before={before:?}");
    println!("case={name} stats_after={after:?}");

    unsafe { q38_ple_close(handle) };
}

fn main() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before UNIX epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "q38-ple-warm-gather-{}-{nonce}.bin",
        std::process::id()
    ));
    let temp = TempFile(path);
    let mut file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&temp.0)
        .expect("create synthetic shard");
    let payload_bytes = ROWS * ROW_BYTES;
    assert!(payload_bytes < 64 * 1024 * 1024);
    let mut payload = vec![0_u8; payload_bytes];
    for (index, byte) in payload.iter_mut().enumerate() {
        *byte = ((index * 29 + index / ROW_BYTES * 17 + 11) & 0xff) as u8;
    }
    file.write_all(&payload).expect("write synthetic shard");
    file.sync_all().expect("sync synthetic shard");

    let sequential: Vec<i64> = (0..ROWS as i64).collect();
    let full_a: Vec<i64> = (0..ROWS)
        .step_by(2)
        .chain((1..ROWS).step_by(2).rev())
        .map(|row_id| row_id as i64)
        .collect();
    let full_b: Vec<i64> = (1..ROWS)
        .step_by(2)
        .chain((0..ROWS).step_by(2).rev())
        .map(|row_id| row_id as i64)
        .collect();
    let subset_a: Vec<i64> = (0..ROWS as i64).step_by(2).collect();
    let subset_b: Vec<i64> = (1..ROWS)
        .step_by(2)
        .rev()
        .map(|row_id| row_id as i64)
        .collect();
    let duplicate_16_head: Vec<i64> = (0..(ROWS / 16) as i64)
        .flat_map(|row_id| std::iter::repeat_n(row_id, 16))
        .collect();

    println!("rows={ROWS} row_bytes={ROW_BYTES} payload_bytes={payload_bytes}");
    run_case(
        "sequential_same_order",
        &temp.0,
        &payload,
        &sequential,
        std::slice::from_ref(&sequential),
    );
    run_case(
        "alternating_full_orders",
        &temp.0,
        &payload,
        &sequential,
        &[full_a, full_b],
    );
    run_case(
        "alternating_subsets",
        &temp.0,
        &payload,
        &sequential,
        &[subset_a, subset_b],
    );
    run_case(
        "duplicate_16_head",
        &temp.0,
        &payload,
        &sequential,
        &[duplicate_16_head],
    );
}
