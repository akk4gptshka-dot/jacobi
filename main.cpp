
 // Computing eigenvectors of a symmetric matrix up to 100x100
 // using the Jacobi method
 // Levshin I.A., 20.2 MKN

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <limits>

using namespace std;

const int MAX_N = 100;
const double EPS = 1e-10;
const int MAX_ITER = 1000000;

using Matrix = vector<vector<double>>;

bool jacobi(Matrix A, Matrix& V, int n) {
    V.assign(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; i++) {
        V[i][i] = 1.0;
    }

    for (int iter = 0; iter < MAX_ITER; iter++) {
        int p = 0, q = 0;
        double maxVal = 0.0;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (abs(A[i][j]) > maxVal) {
                    maxVal = abs(A[i][j]);
                    p = i;
                    q = j;
                }
            }
        }

        if (maxVal < EPS) {
            return true;
        }

        double app = A[p][p];
        double aqq = A[q][q];
        double apq = A[p][q];

        double tau = (aqq - app) / (2.0 * apq);

        double t;
        if (tau >= 0.0) {
            t = 1.0 / (tau + sqrt(1.0 + tau * tau));
        } else {
            t = -1.0 / (-tau + sqrt(1.0 + tau * tau));
        }

        double c = 1.0 / sqrt(1.0 + t * t);
        double s = t * c;

        A[p][p] = app - t * apq;
        A[q][q] = aqq + t * apq;
        A[p][q] = 0.0;
        A[q][p] = 0.0;

        for (int k = 0; k < n; k++) {
            if (k != p && k != q) {
                double akp = A[k][p];
                double akq = A[k][q];

                A[k][p] = c * akp - s * akq;
                A[p][k] = A[k][p];

                A[k][q] = s * akp + c * akq;
                A[q][k] = A[k][q];
            }
        }

        for (int k = 0; k < n; k++) {
            double vkp = V[k][p];
            double vkq = V[k][q];

            V[k][p] = c * vkp - s * vkq;
            V[k][q] = s * vkp + c * vkq;
        }
    }

    return false;
}

int main() {
    int n;

    cout << "Enter matrix size (1..100): ";
    if (!(cin >> n) || n < 1 || n > MAX_N) {
        cout << "Invalid matrix size.\n";
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        cout << "Press Enter to exit...";
        cin.get();
        return 1;
    }

    Matrix A(n, vector<double>(n));

    cout << "Enter matrix elements row by row:\n";

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (!(cin >> A[i][j]) || !isfinite(A[i][j])) {
                cout << "Invalid matrix element.\n";
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                cout << "Press Enter to exit...";
                cin.get();
                return 1;
            }
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double scale = max({
                1.0, abs(A[i][j]), abs(A[j][i])
            });

            if (abs(A[i][j] - A[j][i]) > 1e-12 * scale) {
                cout << "The Jacobi method works only with symmetric matrices.\n";
                cout << "Press Enter to exit...";
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                cin.get();
                return 1;
            }
        }
    }

    Matrix V;

    if (!jacobi(A, V, n)) {
        cout << "Method did not converge.\n";
        cout << "Press Enter to exit...";
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        cin.get();
        return 1;
    }

    cout << "\nComputed eigenvectors (columns of V):\n";
    cout << fixed << setprecision(6);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << setw(12) << V[i][j];
        }
        cout << '\n';
    }

    cout << "\nPress Enter to exit...";
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    cin.get();

    return 0;
}
