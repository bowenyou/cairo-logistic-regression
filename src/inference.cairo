use orion::operators::{
    tensor::{core::{Tensor, TensorTrait}, implementations::impl_tensor_fp::Tensor_fp},
    nn::{core::NNTrait, implementations::impl_nn_fp::NN_fp}
};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType};

fn pred(x: Tensor<FixedType>, theta: Tensor<FixedType>) -> Tensor<FixedType> {
    let weight = x.matmul(@theta);
    let y_pred = NNTrait::sigmoid(@weight);

    return y_pred;
}
