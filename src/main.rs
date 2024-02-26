use ndarray::{arr1, Array1};

#[allow(unused_imports)]
use ndarray_linalg::Lapack;

fn predict(features: &Array1<f64>, model: (f64, f64)) -> Array1<f64> {
    features * model.0 + model.1
}

#[inline(always)]
fn error(prediction: &Array1<f64>, labels: &Array1<f64>) -> Array1<f64> {
    prediction - labels
}

#[inline(always)]
fn dm(features: &Array1<f64>, error: &Array1<f64>) -> f64 {
    2.0_f64 * features.dot(error) / features.len() as f64
}

#[inline(always)]
fn db(error: &Array1<f64>) -> f64 {
    2.0_f64 * error.sum() / error.len() as f64
}

fn linreg(features: &Array1<f64>, labels: &Array1<f64>) -> (f64, f64) {
    let mut slope = 1.0;
    let mut y_int = 0.0;

    // should align within one page but im assuming it won't be that much slower if it didn't
    const EPOCH: usize = 1000;
    const LEARN_RATE: f64 = 0.01;

    for _ in 0..EPOCH {
        let prediction = predict(features, (slope, y_int));
        let error = error(&prediction, labels);
        let dm = dm(features, &error);
        let db = db(&error);
        slope -= LEARN_RATE * dm;
        y_int -= LEARN_RATE * db;
    }

    (slope, y_int)
}

fn main() {
    // sample data
    let hours_studied: Array1<f64> =
        Array1::from(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let exam_scores: Array1<f64> =
        Array1::from(vec![60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0]);

    // getting initial model from the data given
    let (m, b) = linreg(&hours_studied, &exam_scores);

    // given some number of hours studied we should be able to predict what grade the student will get
    let hours = arr1(&[4.25]);
    let predicted_exam_grade = predict(&hours, (m, b));

    println!("predicted grade is {predicted_exam_grade}");
    println!("{m}, {b}");
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    // litterally a test to make sure the ndarray and ndarray-linalg are
    // working properly using intel-mkl-static linalg feats
    #[test]
    pub(crate) fn test_ndarray() {
        use ndarray::arr2;
        // Create two matrices
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);

        // Perform matrix multiplication
        let result = a.dot(&b);

        // Print the result
        println!("Matrix multiplication result:\n{:?}", result);
    }
}
