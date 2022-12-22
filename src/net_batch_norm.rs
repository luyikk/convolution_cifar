use tch::{nn, Tensor};
use tch::nn::{ConvConfig, ModuleT};


#[derive(Debug)]
pub struct NetBatchNorm {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    batch_norm1: nn::BatchNorm,
    batch_norm2: nn::BatchNorm,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl NetBatchNorm {
    pub fn new(vs: &nn::Path) -> Self {
        Self {
            conv1: nn::conv2d(
                vs,
                3,
                16,
                3,
                ConvConfig {
                    padding: 1,
                    ..Default::default()
                },
            ),
            conv2: nn::conv2d(
                vs,
                16,
                8,
                3,
                ConvConfig {
                    padding: 1,
                    ..Default::default()
                },
            ),
            batch_norm1: nn::batch_norm2d(vs, 16, Default::default()),
            batch_norm2: nn::batch_norm2d(vs, 8, Default::default()),
            fc1: nn::linear(vs, 8 * 8 * 8, 32, Default::default()),
            fc2: nn::linear(vs, 32, 2, Default::default()),
        }
    }
}

impl ModuleT for NetBatchNorm {
    #[inline]
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let out = xs.view([-1, 3, 32, 32])
            .apply(&self.conv1)
            .apply_t(&self.batch_norm1, train);
        let out = out.relu() + out;
        let out = out.max_pool2d_default(2)
            .apply(&self.conv2)
            .apply_t(&self.batch_norm2,train);
        let out = out.relu() + out;
        out.max_pool2d_default(2)
            .view([-1,8*8*8])
            .apply(&self.fc1)
            .relu()
            .apply(&self.fc2)

    }
}
