use digitnn::{
    Image,
    idx::{
        IdxData,
        Num,
    },
    Network,
};

fn main() {
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

    for (count, p) in img_vec.iter().zip(img_labels.iter_mut()).enumerate() {
        let (i, j) = p;
        net.run(&i.data_1d);
        i.print();
        println!(
            "\nLabel:\n{}\nOutput:\n{} => {}\n{:?}\n",
            j,
            net.best_match.unwrap(),
            net.surety.unwrap(),
            net.output.as_ref().unwrap()
        );
        println!("{}\n", count+1);
        std::thread::sleep(std::time::Duration::new(2,0));
    }

    println!("Image:\n{}", img_vec[34]);
    println!("Label:\n{}", img_labels[34]);
}
