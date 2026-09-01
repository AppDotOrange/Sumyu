use grad::helper::{dump_vocab_as_rust, make_vocab_from_file};

fn main() {
    let vocab =
        make_vocab_from_file(
            "Datasets/fineweb_v2_sample.txt",
            20_000,
            0,
            Some(&[
                "<EOT>", "<USER>", "<BOT>", "<TOOL>", "<TOOLEND>"
            ]),
        ).unwrap();
    dump_vocab_as_rust(
        &vocab,
        "vocabs/fineweb_vocab3.rs"
    ).unwrap()
}
