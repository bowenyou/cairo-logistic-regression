use array::ArrayTrait;
use orion::operators::tensor::{
    core::{Tensor, TensorTrait, ExtraParams}, implementations::impl_tensor_fp::Tensor_fp
};
use orion::numbers::fixed_point::{
    core::{FixedTrait, FixedType, FixedImpl}, implementations::fp16x16::core::FP16x16Impl
};
fn X_train() -> Tensor<FixedType> {
    let mut shape = ArrayTrait::new();
    shape.append(100);
    shape.append(3);
    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(2, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, true));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, true));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(3, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(3, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, true));
    data.append(FixedTrait::new_unscaled(4, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(3, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, true));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, true));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(2, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, true));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, true));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(0, true));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, true));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, true));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, true));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, true));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, true));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, true));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, true));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, true));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, true));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(0, true));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, true));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(2, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, true));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, true));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(2, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, true));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(2, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, true));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(2, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(2, true));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, true));
    data.append(FixedTrait::new_unscaled(0, true));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, true));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, true));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(1, false));
    data.append(FixedTrait::new_unscaled(0, false));
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
    let tensor = TensorTrait::<FixedType>::new(shape.span(), data.span(), Option::Some(extra));
    return tensor;
}