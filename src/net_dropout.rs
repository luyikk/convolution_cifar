use tch::{nn, Tensor};
use tch::nn::{ConvConfig, ModuleT};


#[derive(Debug)]
pub struct NetDropout {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl NetDropout {
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
            fc1: nn::linear(vs, 8 * 8 * 8, 32, Default::default()),
            fc2: nn::linear(vs, 32, 2, Default::default()),
        }
    }
}

impl ModuleT for NetDropout {
    #[inline]
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.view([-1, 3, 32, 32])
            .apply(&self.conv1)
            .relu()
            .max_pool2d_default(2)
            .dropout(0.4,train)
            .apply(&self.conv2)
            .relu()
            .max_pool2d_default(2)
            .dropout(0.4,train)
            .view([-1, 8 * 8 * 8])
            .apply(&self.fc1)
            .relu()
            .apply(&self.fc2)
    }
}
