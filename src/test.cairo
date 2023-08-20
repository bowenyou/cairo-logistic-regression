use debug::PrintTrait;
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{
    core::{Tensor, TensorTrait, ExtraParams},
    implementations::impl_tensor_fp::{
        Tensor_fp, FixedTypeTensorMul, FixedTypeTensorSub, FixedTypeTensorDiv
    }
};
use orion::numbers::fixed_point::{
    core::{FixedTrait, FixedType, FixedImpl},
    implementations::fp16x16::core::{FP16x16Impl, FP16x16Div, FP16x16PartialOrd}
};

use logistic_regression::{
    generated::{X_train::X_train, Y_train::Y_train, X_test::X_test, Y_test::Y_test},
    train::{train, calculate_loss}, inference::pred
};

#[test]
#[available_gas(99999999999999999)]
fn test() {
    let x_train = X_train();
    'loaded x_train'.print();
    let x_test = X_test();
    'loaded x_test'.print();
    let y_train = Y_train();
    'loaded y_train'.print();
    let y_test = Y_test();
    'loaded y_test'.print();

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };

    let feature_size = *x_train.shape[1];

    let mut zero_array = ArrayTrait::new();

    let mut i = 0_u32;
    loop {
        if i >= feature_size {
            break ();
        }
        zero_array.append(FP16x16Impl::ZERO());
        i += 1;
    };

    let initial_theta = TensorTrait::new(
        shape: array![feature_size].span(), data: zero_array.span(), extra: Option::Some(extra), 
    );

    zero_array = ArrayTrait::new();
    i = 0;
    let train_size = y_train.data.len();
    loop {
        if i >= train_size {
            break ();
        }
        zero_array.append(FixedTrait::new(32768, false)); // naive prediction of 0.5 
        i += 1;
    };

    let initial_y = TensorTrait::new(
        shape: array![train_size].span(), data: zero_array.span(), extra: Option::Some(extra), 
    );

    let initial_loss = calculate_loss(y_train, initial_y);

    let alpha = FixedTrait::new(32768, false); // 655 is 0.01

    let final_theta = train(x_train, y_train, initial_theta, alpha, 100_u32);
    let final_y_pred = pred(x_train, final_theta);

    let final_loss = calculate_loss(y_train, final_y_pred);

    initial_loss.mag.print();
    final_loss.mag.print();
    assert(final_loss < initial_loss, 'no decrease in training loss');
}
