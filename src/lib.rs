use rand::Rng;
use std::fmt;
use std::io::Write;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

pub mod idx;

fn squish(i: f64) -> f64 {
    1.0 / ( 1.0 + (-i).exp() )
}

pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub value: f64,
}

impl Neuron {
    pub fn new_random(len: usize) -> Neuron {
        let mut rng = rand::thread_rng();
        let mut n: Vec<f64> = vec![];
        for _ in 0..len {
            n.push(rng.gen_range(-1.0..1.0));
        }
        let bias = rng.gen_range(-1.0..1.0);
        Neuron {
            weights: n,
            bias,
            value: 0.0,
        }
    }

    pub fn trigger(&mut self, input: &Vec<f64>) -> Result<f64, Error> {
        // println!("Weights\n{:?}\n\n", self.weights);
        if input.len() == self.weights.len() {
            self.value = squish(self.weights.iter().zip(input.iter()).map(|(x, y)| x*y).sum::<f64>() + self.bias);
            Ok(self.value)
        } else {
            Err(Error::WrongSize)
        }
    }
}

pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub values: Vec<f64>,
}

impl Layer {
    pub fn new_random(input_size: usize, output_size: usize) -> Layer {
        let mut n: Vec<Neuron> = vec![];
        for _ in 0..output_size {
            n.push(Neuron::new_random(input_size));
        }
        Layer {
            values: vec![0.0; n.len()],
            neurons: n,
        }
    }

    pub fn run(&mut self, inputs: Vec<f64>) {
        self.values = self.neurons.iter_mut().map( |x| x.trigger(&inputs).unwrap() ).collect();
    }
}

pub struct Network {
    pub layers: Vec<Layer>,
    pub output: Option<Vec<f64>>,
    pub best_match: Option<u8>,
    pub surety: Option<f64>,
}

impl Network {
    pub fn new_random(input_size: usize, layer_sizes: Vec<usize>) -> Network {
        let mut l: Vec<Layer> = vec![];
        l.push(Layer::new_random(input_size, layer_sizes[0]));
        for i in 1..layer_sizes.len() {
            l.push(Layer::new_random(layer_sizes[i-1], layer_sizes[i]));
        }
        Network {
            layers: l,
            output: None,
            best_match: None,
            surety: None,
        }
    }

    pub fn run(&mut self, input: &Vec<u8>) {
        let input: Vec<f64> = input.iter().map(|n| *n as f64).collect();
        self.layers[0].run(input);
        for i in 1..self.layers.len() {
            // println!("{:?}", output);
            let last_layer = self.layers[i-1].values.clone();
            self.layers[i].run(last_layer);
        }
        self.output = Some(self.layers[self.layers.len()-1].values.clone());
        let (surety, best_match) = {
            let mut x = &f64::NEG_INFINITY;
            let mut index = 0;
            for (c, i) in self.output.as_ref().unwrap().iter().enumerate() {
                if i > x {
                    x = i;
                    index = c;
                }
            }
            (Some(*x), Some(index as u8))
        };
        self.surety = surety;
        self.best_match = best_match;
    }
}

#[derive(Debug)]
pub struct Image {
    pub dimensions: (usize, usize),
    pub data: Vec<Vec<u8>>,
    pub data_1d: Vec<u8>,
}

impl Image {
    pub fn from_slice(d: &[idx::Num], x: usize, y: usize) -> Image {
        let mut data: Vec<Vec<u8>> = vec![vec![]; y];
        let mut data_1d: Vec<u8> = vec![];

        for (c, n) in d.iter().enumerate() {
            let i = match n {
                    idx::Num::Unsigned(u) => *u,
                    _ => panic!("Not u8!"),
            };
            data[c/y].push(i);
            data_1d.push(i)
        }
        Image { dimensions: (x, y), data, data_1d }
    }
}

impl Image {
    pub fn print(&self) {
        let mut stdo = StandardStream::stdout(ColorChoice::Always);
        for i in self.data.iter() {
            for j in i.iter() {
                stdo.set_color(
                    ColorSpec::new()
                    .set_bg(Some(Color::Rgb(*j, *j, *j)))
                    .set_fg(Some(Color::Rgb(*j/2, *j/2, *j/2)))
                ).unwrap();
                write!(&mut stdo, "{:2X}", j).unwrap();
            }
            WriteColor::reset(&mut stdo).unwrap();
            write!(&mut stdo, "\n").unwrap();
        }
    }
}

impl fmt::Display for Image {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut out: String = String::from("");
        for i in self.data.iter() {
            for j in i.iter() {
                // if j >= &128 {
                //     out = format!("{}00", out);
                // } else {
                //     out = format!("{}  ", out);
                // }
                out = format!("{}{:2X}", out, j );
            }
            out = format!("{}\n", out);
        }
        write!(f, "{}", out)
    }
}

#[derive(Debug)]
pub enum Error {
    WrongSize
}
