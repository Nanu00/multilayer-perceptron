use std::{fs::File, io::BufReader};
use byteorder::{BigEndian, ReadBytesExt};

#[derive(Debug)]
pub enum Num {
    Unsigned(u8),
    Signed(i8),
    Short(i16),
    Int(i32),
    Float(f32),
    Double(f64),
}

#[derive(Debug)]
pub struct IdxData {
    pub path: String,
    pub magic_number: i32,
    pub data: Vec<Num>,
    pub sizes: Vec<usize>,
    pub total_size: u128,
}

impl IdxData {
    pub fn new(path: impl ToString) -> IdxData {
        let magic_number: i32;
        let mut total_size: u128;
        let mut data: Vec<Num> = vec![];
        let mut sizes: Vec<usize> = vec![];

        let f = File::open(path.to_string()).unwrap();
        let mut reader = BufReader::new(f);

        magic_number = reader.read_i32::<BigEndian>().unwrap();

        for _ in 0..(magic_number%256) {
            sizes.push(reader.read_i32::<BigEndian>().unwrap() as usize)
        }

        total_size = sizes[0] as u128;
        for i in 1..sizes.len() {
            total_size *= sizes[i] as u128;
        }

        match magic_number/256 {
            0x08 => {
                for _ in 0..total_size {
                    data.push(Num::Unsigned(reader.read_u8().unwrap()));
                }
            },
            0x09 => {
                for _ in 0..total_size {
                    data.push(Num::Signed(reader.read_i8().unwrap()));
                }
            },
            0x0B => {
                for _ in 0..total_size {
                    data.push(Num::Short(reader.read_i16::<BigEndian>().unwrap()));
                }
            },
            0x0C => {
                for _ in 0..total_size {
                    data.push(Num::Int(reader.read_i32::<BigEndian>().unwrap()));
                }
            },
            0x0D => {
                for _ in 0..total_size {
                    data.push(Num::Float(reader.read_f32::<BigEndian>().unwrap()));
                }
            },
            0x0E => {
                for _ in 0..total_size {
                    data.push(Num::Double(reader.read_f64::<BigEndian>().unwrap()));
                }
            },
            _ => panic!("Unknown data type"),
        };

        IdxData { path: path.to_string(), magic_number, data, sizes, total_size }
    }
}
