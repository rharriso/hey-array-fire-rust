#[macro_use(af_print)]
extern crate arrayfire as af;

use af::*;

fn main() {
    af::set_device(0);

    af::info();
    println!("Arrayfire version: {:?}", af::get_version());
    let (name, platform, toolkit, compute) = af::device_info();
    print!("Name: {}\nPlatform: {}\nToolkit: {}\nCompute: {}\n", name, platform, toolkit, compute);
    println!("Revision: {}", af::get_revision());

    let num_rows: u64 = 50000;
    let num_cols: u64 = 3000;
    let values: [f32; 3] = [1.0, 2.0, 3.0];
    let indices = af::Array::new(&values, af::Dim4::new(&[3, 1, 1, 1]));

    let dims = Dim4::new(&[num_rows, num_cols, 1, 1]);

    let a = randu::<f32>(dims);
    
    let dimsB = Dim4::new(&[num_cols, 1, 1, 1]);
    let b = randu::<f32>(dimsB);

    //println!("{}", a.dims());
    //println!("{}", b.dims());
    let c = af::matmul(&a, &b, af::MatProp::NONE, af::MatProp::NONE);
    //println!("{}", c.dims());
    
    let dimsD = Dim4::new(&[num_rows, 1, 1, 1]);
    let d = randu::<f32>(dimsD);
    let e = af::matmul(&c, &d, af::MatProp::TRANS, af::MatProp::NONE);
    //println!("{}", e.dims());
    af_print!("Result of mult", e);
}
