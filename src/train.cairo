use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{
    core::{Tensor, TensorTrait, ExtraParams},
    implementations::impl_tensor_fp::{
        Tensor_fp, FixedTypeTensorMul, FixedTypeTensorSub, FixedTypeTensorDiv
    }
};
use orion::numbers::fixed_point::{
    core::{FixedTrait, FixedType, FixedImpl},
    implementations::fp16x16::core::{FP16x16Impl, FP16x16Div}
};
use logistic_regression::inference::pred;
use debug::PrintTrait;


fn calculate_loss(y: Tensor<FixedType>, y_pred: Tensor<FixedType>) -> FixedType {
    let tensor_size = FP16x16Impl::new_unscaled(y.data.len(), false);

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
    let one = TensorTrait::new(
        shape: array![1].span(),
        data: array![FixedTrait::new(65536, false)].span(),
        extra: Option::Some(extra),
    );

    let neg_one = TensorTrait::new(
        shape: array![1].span(),
        data: array![FixedTrait::new(65536, true)].span(),
        extra: Option::Some(extra),
    );

    let res = (neg_one * y * y_pred.log()) - (one - y) * (one - y_pred).log();

    let cumsum = res.cumsum(0, Option::None(()), Option::None(()));
    let sum = cumsum.data[res.data.len() - 1];
    let mean = FP16x16Div::div(*sum, tensor_size);

    return mean;
}

fn calculate_gradient(
    x: Tensor<FixedType>, y: Tensor<FixedType>, y_pred: Tensor<FixedType>
) -> Tensor<FixedType> {
    let tensor_size = FP16x16Impl::new_unscaled(y.data.len(), false);
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };

    let scale = TensorTrait::new(
        shape: array![1].span(), data: array![tensor_size].span(), extra: Option::Some(extra), 
    );

    let x_transpose = x.transpose(axes: array![1, 0].span());

    let gradient = x_transpose.matmul(@(y_pred - y)) / scale;

    return gradient;
}

fn train_step(
    x: Tensor<FixedType>, y: Tensor<FixedType>, theta: Tensor<FixedType>, alpha: FixedType
) -> Tensor<FixedType> {
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
    let alpha_tensor = TensorTrait::new(
        shape: array![1].span(), data: array![alpha].span(), extra: Option::Some(extra), 
    );

    let y_pred = pred(x, theta).reshape(y.shape);

    let gradient = calculate_gradient(x, y, y_pred);
    let update = (gradient * alpha_tensor);

    let new_theta = theta - update;

    return new_theta;
}

fn train(
    x: Tensor<FixedType>,
    y: Tensor<FixedType>,
    init_theta: Tensor<FixedType>,
    alpha: FixedType,
    n_iters: u32
) -> Tensor<FixedType> {
    let mut i = 0;
    let mut theta = init_theta;

    loop {
        if i >= n_iters {
            break ();
        }
        theta = train_step(x, y, theta, alpha);
        i += 1;
    };

    return theta;
}
