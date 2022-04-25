use digitnn::{Image, idx::{IdxData, Num}, Network};
use std::fs::OpenOptions;
use ron::{ser::to_writer, de::from_reader};
use clap::{Arg, Command};
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};

fn train(output_path: &str) {
    let mut net = Network::new_random(784, vec![200, 75, 10]);

    let idx_imgs = IdxData::new("./dataset/train-images-idx3-ubyte");
    let idx_labels = IdxData::new("./dataset/train-labels-idx1-ubyte");
    let res = idx_imgs.sizes[1]*idx_imgs.sizes[2];

    let mut img_vec: Vec<Image> = Vec::new();
    let mut img_labels: Vec<u8> = Vec::new();

    for i in 0..idx_imgs.sizes[0] {
        img_vec.push(Image::from_slice(&idx_imgs.data[(res*i)..(res*(i+1))], idx_imgs.sizes[1], idx_imgs.sizes[2]))
    }
    
    for i in idx_labels.data.iter() {
        img_labels.push(
            match i {
                Num::Unsigned(n) => *n,
                _ => panic!("Not unsigned!"),
            }
        )
    }

    let mut expected: Vec<f64>;

    let pb = ProgressBar::new(idx_imgs.sizes[0] as u64)
        .with_style(
            ProgressStyle::default_bar()
                    .template("{msg} |{wide_bar}| ETA: {eta_precise} {pos}/{len}")
                    )
        .with_message("Training");

    for p in
        img_vec
        .iter()
        .zip(img_labels.iter_mut())
        .progress_with(pb)
        {
        let (i, j) = p;

        net.run(&i.data_1d);

        expected = vec![0.0; 9];
        expected.insert(*j as usize, 1.0);

        let wd: Vec<Vec<Vec<f64>>>;
        let bd: Vec<Vec<f64>>;
        let x = net.deltas(&expected);
        wd = x.0;
        bd = x.1;

        net.apply_deltas(wd, bd, 0.1);
    }
    
    let f = OpenOptions::new()
                .write(true)
                .truncate(true)
                .create(true)
                .open(output_path)
                .expect("Error opening file");

    to_writer(f, &net).expect("Error writing to file");
    println!("Neural network dumped to {}", output_path);
}

fn test(input_path: &str) {
    let f = OpenOptions::new()
                .read(true)
                .open(input_path)
                .expect("Error opening file");

    let mut net: Network = match from_reader(f){
        Ok(n) => n,
        Err(e) => {
            println!("Failed to load file: {}", e);

            std::process::exit(1);
        }
    };
    let unseen_idx_imgs = IdxData::new("./dataset/t10k-images-idx3-ubyte");
    let unseen_idx_labels = IdxData::new("./dataset/t10k-labels-idx1-ubyte");

    let mut unseen_img_vec: Vec<Image> = Vec::new();
    let mut unseen_img_labels: Vec<u8> = Vec::new();

    let res = unseen_idx_imgs.sizes[1]*unseen_idx_imgs.sizes[2];

    for i in 0..unseen_idx_imgs.sizes[0] {
        unseen_img_vec.push(Image::from_slice(&unseen_idx_imgs.data[(res*i)..(res*(i+1))], unseen_idx_imgs.sizes[1], unseen_idx_imgs.sizes[2]))
    }
    
    for i in unseen_idx_labels.data.iter() {
        unseen_img_labels.push(
            match i {
                Num::Unsigned(n) => *n,
                _ => panic!("Not unsigned!"),
            }
        )
    }

    let mut correct = 0;

    let pb = ProgressBar::new(unseen_idx_imgs.sizes[0] as u64)
        .with_style(ProgressStyle::default_bar()
                    .template("{msg} |{wide_bar}| ETA: {eta_precise} {pos}/{len}")
                    // .progress_chars("#- ")
                    )
        .with_message("Testing");

    for p in 
        unseen_img_vec
            .iter()
            .zip(unseen_img_labels.iter_mut())
            .progress_with(pb)
    {
        let (i, j) = p;

        net.run(&i.data_1d);

        if net.best_match.unwrap() == *j {
            correct = correct + 1;
        }
    }
    println!("Accuracy: {}%", (correct*100) as f64/(unseen_img_labels.len() as f64));
}

fn main() {
    let args = Command::new("digitnn")
    .author("Nanu00 <github.com/Nanu00>")
    .about("Simple neural network")
    .subcommand(
            Command::new("train")
            .about("Train the network")
            .arg(
                    Arg::new("output")
                    .short('o')
                    .long("output")
                    .takes_value(true)
                    .value_name("FILE")
                    .help("Output file")
                    .required(true)
            )
    )
    .subcommand(
            Command::new("test")
            .about("Test the network")
            .arg(
                Arg::new("network")
                .short('i')
                .long("network")
                .takes_value(true)
                .value_name("FILE")
                .help("Input file (RON format)")
                .required(true)
            )
    );

    let arg_m = args.get_matches();

    match arg_m.subcommand() {
        Some(("train", sub_m)) => { train(sub_m.value_of("output").unwrap()) }
        Some(("test", sub_m)) => { test(sub_m.value_of("network").unwrap()) }
        _ => {
            println!("No command provided!");
            std::process::exit(0);
        }
    }
}
