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

    let mut expected: Vec<f64>;

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

        expected = vec![0.0; 9];
        expected.insert(*j as usize, 1.0);

        println!("Cost: {}\n", net.cost(&expected));
        println!("{}\n", count+1);

        // println!("Deltas: {:?}\n", net.deltas(&expected));
        let wd: Vec<Vec<Vec<f64>>>;
        let bd: Vec<Vec<f64>>;
        let x = net.deltas(&expected);
        wd = x.0;
        bd = x.1;

        // println!("{:?}", wd[2][0]);

        net.apply_deltas(wd, bd, 0.1);

        // std::thread::sleep(std::time::Duration::new(10,0));
    }


    let unseen_idx_imgs = IdxData::new("./dataset/t10k-images-idx3-ubyte");
    let unseen_idx_labels = IdxData::new("./dataset/t10k-labels-idx1-ubyte");

    let mut unseen_img_vec: Vec<Image> = Vec::new();
    let mut unseen_img_labels: Vec<u8> = Vec::new();

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

    for (count, p) in unseen_img_vec.iter().zip(unseen_img_labels.iter_mut()).enumerate() {
        let (i, j) = p;

        net.run(&i.data_1d);

        if net.best_match.unwrap() == *j {
            correct = correct + 1;
        }

        i.print();
        println!(
            "\nLabel:\n{}\nOutput:\n{} => {}\n{:?}\n",
            j,
            net.best_match.unwrap(),
            net.surety.unwrap(),
            net.output.as_ref().unwrap()
        );

        expected = vec![0.0; 9];
        expected.insert(*j as usize, 1.0);

        println!("Cost: {}\n", net.cost(&expected));
        println!("{}\n", count+1);
        println!("\n\nAccuracy: {}%", (correct*100)/(count+1));
        // std::thread::sleep(std::time::Duration::new(2,0));
    }

}
