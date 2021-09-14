use rand::Rng;
use std::fmt;
use std::io::Write;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

pub mod idx;

fn squish(i: f64) -> f64 {
    1.0 / ( 1.0 + (-i).exp() )
}

pub struct Neuron {
    pub input: Vec<f64>,
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
            input: vec![],
            weights: n,
            bias,
            value: 0.0,
        }
    }

    pub fn trigger(&mut self, input: &Vec<f64>) -> Result<f64, Error> {
        // println!("Weights\n{:?}\n\n", self.weights);
        if input.len() == self.weights.len() {
            self.input = input.clone();
            self.value = squish(self.weights.iter().zip(input.iter()).map(|(x, y)| x*y).sum::<f64>() + self.bias);
            Ok(self.value)
        } else {
            Err(Error::WrongSize)
        }
    }

    pub fn apply_deltas(&mut self, wd: &Vec<f64>, bd: f64, lr: f64) {
        self.weights.iter_mut().zip(wd.iter()).for_each(|(w, d)| *w -= d * lr );
        self.bias -= bd * lr;
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

    pub fn apply_deltas(&mut self, wd: &Vec<Vec<f64>>, bd: &Vec<f64>, lr: f64) {
        // println!("{}-{}-{}", wd.len(), bd.len(), self.neurons.len());
        for (j, n) in self.neurons.iter_mut().enumerate() {
            n.apply_deltas(&wd[j], bd[j], lr);
        }
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

    // pub fn bp(&mut self, expected: &Vec<u8>) {
        
    // }
    
    pub fn cost(&self, expected: &Vec<f64>) -> f64 {
        self.output.as_ref().unwrap().iter()
            .zip(expected.iter())
            .map( |(a, b)| (b - a).powf(2.0) )
            .sum()
    }

    pub fn deltas(&self, expected: &Vec<f64>) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
        let mut v2d_dc_dw: Vec<Vec<Vec<f64>>> = vec![];
        let mut v2d_dc_db: Vec<Vec<f64>> = vec![vec![]; self.layers.len()];

        let mut v_dc_da: Vec<f64> = self.output.as_ref().unwrap().iter()
            .zip(expected.iter())
            .map( | ( a, y ) | 2.0 * ( a - y ) )
            .collect();
        // println!("{:?}", v_dc_da);
        let mut v_dc_da_next: Vec<f64> = vec![];

        for (lnum, l) in self.layers.iter().enumerate().rev() {
            let mut v1d_dc_dw: Vec<Vec<f64>> = vec![vec![]; l.neurons.len()];
            if lnum != 0 {
                v_dc_da_next = vec![0.0; self.layers[lnum-1].neurons.len()];
            }
            for (count, n) in l.neurons.iter().enumerate() {
                let z = &n.value;
                let dc_dz = 0.2 * v_dc_da[count];
                // let dc_dz = z * ( 1.0 - z ) * v_dc_da[count];
                // if d_squish(z).is_finite() {
                //     dc_dz = d_squish(z) * v_dc_da[count];
                // }
                v2d_dc_db[lnum].push(dc_dz);
                for i in n.input.iter() {
                    v1d_dc_dw[count].push(dc_dz * i);
                }
                // println!("{:?}", v1d_dc_dw);
                // std::thread::sleep(std::time::Duration::new(5,0));
                for (i, w) in v_dc_da_next.iter_mut().zip(n.weights.iter()) {
                    *i += dc_dz * w;
                }
            }
            v2d_dc_dw.push(v1d_dc_dw);
            v_dc_da = v_dc_da_next.clone();
            // println!("{:?}", v_dc_da_next);
            // std::thread::sleep(std::time::Duration::new(2,0));
        }
        // println!("{:?}", v2d_dc_dw);
        // std::thread::sleep(std::time::Duration::new(10,0));
        (v2d_dc_dw, v2d_dc_db)
    }

    pub fn apply_deltas(&mut self, wd: Vec<Vec<Vec<f64>>>, bd: Vec<Vec<f64>>, lr: f64) {
        let bd: Vec<Vec<f64>> = bd.into_iter().rev().collect();
        for (j, l) in self.layers.iter_mut().rev().enumerate() {
            l.apply_deltas(&wd[j], &bd[j], lr);
        }
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
