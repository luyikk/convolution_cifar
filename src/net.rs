use tch::nn::{ConvConfig, ModuleT};
use tch::{nn, Tensor};

#[derive(Debug)]
pub struct Net {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Net {
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

impl ModuleT for Net {
    #[inline]
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        xs.view([-1, 3, 32, 32])
            .apply(&self.conv1)
            .tanh()
            .max_pool2d_default(2)
            .apply(&self.conv2)
            .tanh()
            .max_pool2d_default(2)
            .view([-1, 8 * 8 * 8])
            .apply(&self.fc1)
            .tanh()
            .apply(&self.fc2)
    }
}
