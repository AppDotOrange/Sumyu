# Sumyu

Sumyu is a small machine learning framework written in Rust.

It started as a from-scratch neural-network project and is mainly used for experimenting with **small feed-forward language models**. The goal is not to build another giant ML framework. It's to have something small enough to understand, modify, and run on a CPU.

The language-model side of Sumyu currently uses embeddings and fully-connected layers rather than Transformers.

## What it has

* Feed-forward neural networks
* Automatic differentiation
* Configurable network architectures
* Mini-batch training
* Token embeddings
* Text tokenization
* Vocabulary generation from text
* Predefined vocabularies for some datasets
* `.sumyu` model files
* Text generation
* Saving and loading trained models
* Continuing training from saved models

There are also some simpler experiments in the project, such as an XOR dataset.

## Language models

A Sumyu language model predicts the next token from a fixed context.

A simplified view is:

```text
text
 │
 ▼
tokenizer
 │
 ▼
tokens
 │
 ▼
embeddings
 │
 ▼
fully-connected layers
 │
 ▼
next-token prediction
```

For example, you can make a model with a 16-token context, 32-dimensional embeddings, and two hidden layers:

use grad::fnn_lm::LM;
use grad::helper;

fn main() {
    let vocab = helper::ml_200_tok_vocab_v3();

    let mut model = LM::new(
        16,         // context length
        vocab,
        &[128, 64], // hidden layers
        32,         // embedding size
    );

    model.params();
}

The architecture is deliberately simple. There is currently no attention mechanism.

## Training

A text corpus can be loaded into the model and used for training:

use grad::fnn_lm::LM;
use grad::helper;
use std::fs;

fn main() {
    let text = fs::read_to_string("corpus.txt")
        .expect("Couldn't read corpus.txt");

    let vocab = helper::ml_200_tok_vocab_v3();

    let mut model = LM::new(
        16,
        vocab,
        &[128, 64],
        32,
    );

    model.load_corpus(&text);

    model.train_options(
        0.01, // learning rate
        100,  // epochs
        32,   // batch size
        0,    // max batches per epoch (0 = unlimited)
    );

    model.train();
}

The training options are:

| Option        | Meaning                                       |
| ------------- | --------------------------------------------- |
| Learning rate | How large each parameter update is            |
| Epochs        | Number of passes through the training process |
| Batch size    | Number of samples processed in a batch        |
| Max batches   | Maximum batches per epoch; `0` means no limit |

## Building a vocabulary

Sumyu doesn't require you to use one fixed vocabulary.

You can generate one from your own text with `make_vocab`:

use grad::helper;
use std::fs;

fn main() {
    let text = fs::read_to_string("corpus.txt")
        .expect("Couldn't read corpus.txt");

    let vocab = helper::make_vocab(
        &text,
        200,
        0,
    );

    println!("Vocabulary size: {}", vocab.len());
}

`make_vocab` starts with the characters found in the dataset and repeatedly adds frequently occurring adjacent token pairs.

For example, a vocabulary might gradually learn tokens such as:

t
h
e
...
th
he
the

This is similar in spirit to BPE-style vocabulary construction, although Sumyu's algorithm is its own simpler implementation rather than a standard BPE implementation.

The `reserved_token_num` argument can be used to leave space at the beginning of the vocabulary for special tokens.

## Predefined vocabularies

The helper module contains several vocabularies that were made for particular experiments.

For example:

helper::char_level_vocab_v1()
helper::general_vocab_v1()
helper::token_vocab_v1()
helper::ml_200_tok_vocab_v3()
helper::poke_v1()
helper::tale_v1()
helper::recipe_v1()

Some of these are character-level vocabularies, while others contain larger tokens learned from particular datasets.

The dataset-specific vocabularies are mostly there because they worked well for the experiments they were created for. They aren't meant to be universal tokenizers.

## Model configuration

Instead of constructing a model manually, you can also create a `Config`.

use grad::helper;

fn main() {
    let config = helper::Config::new(
        0.01,                     // learning rate
        32,                       // batch size
        0,                        // max batches per epoch
        helper::general_vocab_v1(),
        16,                       // context length
        32,                       // embedding size
        &[128, 64],               // hidden layers
        100,                      // epochs
    );

    println!("Learning rate: {}", config.lr);
}

This is useful when experimenting with different model configurations.

Sumyu also contains helper configurations for some of the models used during development.

## Saving models

Once you've trained a model, you can save it as a `.sumyu` file:

model.save(
    "my_model.sumyu",
    "A model trained on my own dataset.",
);

The `.sumyu` format stores the trained model along with a description.

This makes it possible to train a model once and then move the resulting file somewhere else for inference or further training.

## Loading models

Saved models can be loaded again:

use grad::fnn_lm::LM;

fn main() {
    let (description, model) =
        LM::load("my_model.sumyu");

    println!("{}", description);

    model.params();
}

A loaded model can also be trained further:

let (description, mut model) =
    LM::load("my_model.sumyu");

model.load_corpus(&text);

model.train_options(
    0.005,
    50,
    32,
    0,
);

model.train();

model.save(
    "my_model_updated.sumyu",
    &description,
);

## Text generation

After training, language models can generate text:

let output = model.generate(
    "Once upon a time".to_string(),
    200,
    0.7,
);

println!("{}", output);

The three arguments are:

prompt
number of tokens to generate
temperature

Temperature controls how much randomness is used when choosing the next token.

## Other experiments

Sumyu isn't limited to one particular dataset or model.

The repository contains configurations and vocabularies for experiments involving:

* Source code
* Pokémon descriptions
* Fairy tales
* Recipes
* Small general text datasets

For example, the same language-model code can be configured with a Pokémon-specific vocabulary and trained on a Pokédex, or configured with a different vocabulary and trained on stories.

There is also an XOR dataset helper:

let dataset = helper::xor();

which provides the classic four XOR input/output pairs.

## Small models

One of the main reasons Sumyu exists is to experiment with **small models**.

The models aren't intended to compete with large language models. They're much smaller and much more limited.

That's part of the point.

A model with a few hundred thousand parameters can be trained on a normal CPU and still learn interesting patterns from a specialized dataset.

This also makes the resulting models easy to experiment with and distribute. Some models produced during development are only hundreds of kilobytes.

## Why not Transformers?

Because that's not what this project is about.

Transformers are extremely useful, but Sumyu is an experiment in seeing what can be done with simpler architectures.

A feed-forward language model has obvious limitations:

* The context is fixed.
* There is no attention mechanism.
* It doesn't scale like modern Transformer LMs.
* Small models have very limited capacity.

But the architecture is also relatively straightforward, which makes it fun to experiment with.

## Building

You'll need a Rust toolchain and Cargo.

Build the project with:

cargo build --release

For running the optimized build:

cargo run --release


## Using your own dataset

A basic workflow is:

1. Prepare a text corpus
          ↓
2. Create or choose a vocabulary
          ↓
3. Configure a model
          ↓
4. Train it
          ↓
5. Save it as .sumyu
          ↓
6. Load it later
          ↓
7. Generate text or continue training

Your dataset does not have to be one of the datasets used in this repository. The point of the framework is to experiment with your own.

Of course, you are responsible for having the appropriate rights to use any dataset you train on.

## Project status

Sumyu is experimental software.

The API, model format, and internal implementation may change.

It is probably not the right choice if you need a production-grade machine-learning framework.

If you want to build a tiny language model, poke around with neural networks, or see how far you can get without a Transformer, that's what Sumyu is for.

## License

Sumyu is distributed under the **Sumyu License**.

The Sumyu software itself may not be sold, licensed for a fee, or incorporated into a commercial product or service without permission from the copyright holder.

Models trained using Sumyu may be sold, licensed, distributed, and used commercially.

See LICENSE for the complete terms.
