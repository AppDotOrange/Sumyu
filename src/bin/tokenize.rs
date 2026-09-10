use grad::helper::{dump_vocab_as_rust, make_vocab_from_file};

fn main() {
    let vocab =
        make_vocab_from_file(
            "Datasets/pokedex.txt",
            200,
            0,
            None,
        ).unwrap();
    println!("{:?}", vocab)
    /*
    dump_vocab_as_rust(
        &vocab,
        "vocabs/fineweb_vocab3.rs"
    ).unwrap()
     */
}
