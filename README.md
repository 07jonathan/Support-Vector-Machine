Code tersebut adalah implementasi sederhana dari Support Vector Machine (SVM) menggunakan metode Gradient Descent untuk optimasi. Berikut adalah penjelasan singkat mengenai kode tersebut:

    Library yang digunakan:
        import numpy as np: Digunakan untuk operasi numerik dan array.
        import pandas as pd: Digunakan untuk manipulasi dan analisis data dalam bentuk dataframe.

    Data training dan testing:
        X_train: Array yang berisi fitur-fitur dari data training.
        y_train: Array yang berisi label (kelas) dari data training.
        X_test: Array yang berisi fitur-fitur dari data testing yang akan diprediksi.

    Kelas SVM:
        __init__: Menginisialisasi hyperparameter seperti learning rate (lr) dan jumlah epochs (epochs), serta parameter SVM seperti weights dan bias.
        fit: Metode untuk melatih model SVM menggunakan Gradient Descent. Mengiterasi sebanyak epochs untuk mengoptimalkan weights dan bias.
        predict: Metode untuk melakukan prediksi kelas pada data testing berdasarkan weights dan bias yang telah ditentukan.

    Inisialisasi dan training model SVM:
        model = SVM(): Membuat objek SVM.
        model.fit(X_train, y_train): Melatih model SVM menggunakan data training (X_train dan y_train).

    Prediksi kelas untuk data testing:
        predictions = model.predict(X_test): Menggunakan model yang telah dilatih untuk memprediksi kelas dari data testing (X_test).

    Menampilkan data dalam bentuk tabel:
        df_train: Menampilkan data training dalam bentuk dataframe pandas.
        df_test: Menampilkan data testing dalam bentuk dataframe pandas, dengan menambahkan kolom Feature 2.

    Menampilkan hasil prediksi:
        Hasil prediksi untuk setiap data testing ditampilkan dalam format yang terstruktur.