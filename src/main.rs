mod net;
mod net_dropout;
mod net_batch_norm;

use anyhow::Result;

use crate::net::Net;
use tch::data::Iter2;
use tch::nn::{ModuleT, Optimizer, OptimizerConfig, VarStore};
use tch::{nn, Device, IndexOp, Kind, Tensor};
use crate::net_batch_norm::NetBatchNorm;
use crate::net_dropout::NetDropout;

#[allow(dead_code)]
fn show_image(img: &Tensor) -> Result<()> {
    let mut image_test = image::RgbImage::new(32, 32);

    for x in 0..32 {
        for y in 0..32 {
            let r: u8 = (f32::from(img.i((0, y, x))) * 255.) as u8;
            let g: u8 = (f32::from(img.i((1, y, x))) * 255.) as u8;
            let b: u8 = (f32::from(img.i((2, y, x))) * 255.) as u8;
            imageproc::drawing::draw_cross_mut(
                &mut image_test,
                image::Rgb([r, g, b]),
                x as i32,
                y as i32,
            );
        }
    }

    let image_file_path = tempfile::Builder::new()
        .suffix(".bmp")
        .tempfile()?
        .into_temp_path();

    image_test.save(&image_file_path)?;
    open::that(&image_file_path)?;
    std::thread::sleep(std::time::Duration::from_secs(5));
    Ok(())
}

#[inline]
fn load_data(vs: &VarStore) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let tensor_cifar_dataset = tch::vision::cifar::load_dir("./data")?;
    let mean = Tensor::of_slice(&[0.4914, 0.4822, 0.4465]).view((3, 1, 1));
    let std = Tensor::of_slice(&[0.2470, 0.2435, 0.2616]).view((3, 1, 1));
    let train_data = tensor_cifar_dataset
        .train_iter(1)
        .filter_map(|(img, lable)| {
            if lable == Tensor::of_slice(&[0]) {
                Some((
                    img.g_sub(&mean).g_div(&std).view([3, 32, 32]),
                    Tensor::of_slice(&[0]),
                ))
            } else if lable == Tensor::of_slice(&[2]) {
                Some((
                    img.g_sub(&mean).g_div(&std).view([3, 32, 32]),
                    Tensor::of_slice(&[1]),
                ))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let var_data = tensor_cifar_dataset
        .test_iter(1)
        .filter_map(|(img, lable)| {
            if lable == Tensor::of_slice(&[0]) {
                Some((
                    img.g_sub(&mean).g_div(&std).view([3, 32, 32]),
                    Tensor::of_slice(&[0]),
                ))
            } else if lable == Tensor::of_slice(&[2]) {
                Some((
                    img.g_sub(&mean).g_div(&std).view([3, 32, 32]),
                    Tensor::of_slice(&[1]),
                ))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let train_img_tensor = Tensor::zeros(
        &[train_data.len() as i64, 3, 32, 32],
        (Kind::Float, vs.device()),
    );
    let train_label_tensor = Tensor::zeros(&[train_data.len() as i64], (Kind::Int64, vs.device()));

    for (i, (img, label)) in train_data.iter().enumerate() {
        train_img_tensor.i(i as i64).copy_(img);
        train_label_tensor.i(i as i64).copy_(&label.view([]));
    }

    let var_img_tensor = Tensor::zeros(
        &[var_data.len() as i64, 3, 32, 32],
        (Kind::Float, vs.device()),
    );
    let var_label_tensor = Tensor::zeros(&[var_data.len() as i64], (Kind::Int64, vs.device()));

    for (i, (img, label)) in var_data.iter().enumerate() {
        var_img_tensor.i(i as i64).copy_(img);
        var_label_tensor.i(i as i64).copy_(&label.view([]));
    }

    Ok((
        train_img_tensor,
        train_label_tensor,
        var_img_tensor,
        var_label_tensor,
    ))
}

#[inline]
fn training_loop<T: ModuleT>(
    epoch: u64,
    device: Device,
    net: &T,
    train_data: &Tensor,
    train_labels: &Tensor,
    opt: &mut Optimizer,
) {
    for i in 1..=epoch {
        let mut count = 0f64;
        let mut loss_train = 0f64;
        for (label, img) in Iter2::new(train_labels, train_data, 1024)
            .to_device(device)
            .shuffle()
        {
            let loss = net.forward_t(&img, true).cross_entropy_for_logits(&label); //.cross_entropy_loss::<Tensor>(&label, None, Reduction::Mean, -100, 0f64);
            opt.backward_step(&loss);
            let size = img.size()[0] as f64;
            loss_train += f64::from(loss) * size;
            count += size;
        }
        if i == 1 || i % 10 == 0 {
            println!(
                "{} epoch:{i},Training loss {}",
                chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
                loss_train / count
            );
        }
    }
}

fn main() -> Result<()> {
    let vs = VarStore::new(Device::cuda_if_available());

    let (train_img_tensor, train_label_tensor, var_img_tensor, var_label_tensor) = load_data(&vs)?;

    // show_image(&train_img_tensor.i(0))?;

    let net = Net::new(&vs.root());
    let mut opt = nn::Adam {
        wd: 0.001,
        ..Default::default()
    }
    .build(&vs, 3e-3)?;

    training_loop(
        100,
        vs.device(),
        &net,
        &train_img_tensor,
        &train_label_tensor,
        &mut opt,
    );

    println!(
        "net Accuracy train:{}",
        net.batch_accuracy_for_logits(&train_img_tensor, &train_label_tensor, vs.device(), 1024)
    );
    println!(
        "net Accuracy var:{}",
        net.batch_accuracy_for_logits(&var_img_tensor, &var_label_tensor, vs.device(), 1024)
    );

    let net_dropout = NetDropout::new(&vs.root());

    training_loop(
        100,
        vs.device(),
        &net_dropout,
        &train_img_tensor,
        &train_label_tensor,
        &mut opt,
    );

    println!(
        "dropout Accuracy train:{}",
        net_dropout.batch_accuracy_for_logits(&train_img_tensor, &train_label_tensor, vs.device(), 1024)
    );
    println!(
        "dropout Accuracy var:{}",
        net_dropout.batch_accuracy_for_logits(&var_img_tensor, &var_label_tensor, vs.device(), 1024)
    );

    let net_batch_norm = NetBatchNorm::new(&vs.root());

    training_loop(
        100,
        vs.device(),
        &net_batch_norm,
        &train_img_tensor,
        &train_label_tensor,
        &mut opt,
    );

    println!(
        "net_batch_norm Accuracy train:{}",
        net_batch_norm.batch_accuracy_for_logits(&train_img_tensor, &train_label_tensor, vs.device(), 1024)
    );
    println!(
        "net_batch_norm Accuracy var:{}",
        net_batch_norm.batch_accuracy_for_logits(&var_img_tensor, &var_label_tensor, vs.device(), 1024)
    );


    Ok(())
}
