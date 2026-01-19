// Utility random functions for Windows/MSVC compatibility
#include <random>
inline int randint(int a, int b) {
    static std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dist(a, b);
    return dist(gen);
}
inline int randint_except(int a, int b, int except) {
    int r;
    do {
        r = randint(a, b);
    } while (r == except);
    return r;
}
inline double frand(double a, double b) {
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> dist(a, b);
    return dist(gen);
}
using namespace std;
using namespace Eigen;

std::random_device rd_h;
std::mt19937 gen_h(rd_h());
normal_distribution<double> norm_dist_h(0,1);
uniform_real_distribution<double> uniform_h(0.0, 1.0);


// Windows-compatible mkpath using _mkdir from <direct.h>
#include <direct.h>
bool mkpath(std::string path) {
    bool bSuccess = false;
    int nRC = _mkdir(path.c_str());
    if (nRC == -1) {
        switch (errno) {
        case ENOENT:
            // parent didn't exist, try to create it
            if (mkpath(path.substr(0, path.find_last_of("/\\"))))
                // Now, try to create again.
                bSuccess = 0 == _mkdir(path.c_str());
            else
                bSuccess = false;
            break;
        case EEXIST:
            // Done!
            bSuccess = true;
            break;
        default:
            bSuccess = false;
            break;
        }
    }
    else
        bSuccess = true;
    return bSuccess;
}


template<typename Num_type>
std::vector<std::vector<Num_type> >
Getdata(std::string filename, std::size_t ignore_line_num = 0){
    std::ifstream reading_file;
    reading_file.open(filename, std::ios::in);
    if(!reading_file)
        throw std::invalid_argument(filename + std::string("could not be opened."));
    
    std::string reading_line_buffer;
    
    //Ignore first ignore_line_num lines
    for(std::size_t line = 0; line < ignore_line_num; line++){
        getline(reading_file,reading_line_buffer);
        if(reading_file.eof()) break;
    }
        
    Num_type num;
    char comma;

    std::vector<std::vector<Num_type> > data;
        
    while(std::getline(reading_file, reading_line_buffer)){
        if(reading_line_buffer.size() == 0) break;
    std::vector<Num_type> temp_data;
    std::istringstream is(reading_line_buffer);
    while(is >> num){
        temp_data.push_back(num);
        is >> comma;
    }
    data.push_back(temp_data);
    }
    return data;
};


void Getmatrix(string filename, vector<vector<double> >& mat){
    cout << "reading " << filename << endl;
    mat= Getdata<double>(filename);
    
};


void Getmatrix(string filename, vector<vector<int> >& mat){
    cout << "reading " << filename << endl;
    mat= Getdata<int>(filename);
};


void printmat(vector<vector<double> >& mat, int size){
    for(int i =0;i<size;i++){
        for(int j=0;j<size;j++){
            cout<<mat[i][j]<<" ";
        }
        cout<<endl;
    }
};


void matinv(Eigen::MatrixXd& mat, Eigen::MatrixXd& matinv){
    matinv=mat.inverse();
};





void normalize_A(int n, Eigen::MatrixXd& A){

    for(int i = 0; i <n;i++){
        double sum=0;
        for (int j = 0; j <n;j++){
           sum+=A(i,j);
        }
         for (int j = 0; j <n;j++){
           A(i,j)*=1.0/sum;
        }
     }
}

////////////////////////////////////////////////////////////////

void check_A_nonneg(int n, Eigen::MatrixXd& A){
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            if(A(i,j)<0)cout<<"ERROR: Aij<0"<<endl;
        }
    }
}


void sym_A(int n, Eigen::MatrixXd& A){
    MatrixXd A_sym(n,n);
    for(int i=0; i<n; i++){for(int j=0; j<n;j++){A_sym(i,j) = 0.5*(A(i,j)+A(j,i));}}//This can break the normalization condition 

    // Normalize A by modifying the diagonal components, which keeps A symmetric.
    for(int i=0; i<n; i++){
        double sum=0;
        for (int j=0; j<n; j++){
            if(i!=j){sum+=A_sym(i,j);}
        }
        A_sym(i,i) = 1- sum;
    }

    A=A_sym;
}


void calc_Pi(int n, Eigen::MatrixXd& A,  vector<double> & Pi, string mode){
    Eigen::EigenSolver<MatrixXd> s(A.transpose()); // the instance s(A) includes the eigensystem
    // std::cout << A << std::endl;
    // std::cout << "eigenvalues:" << std::endl;
    // std::cout << s.eigenvalues() << std::endl;
    // std::cout << "eigenvectors=" << std::endl;
    // std::cout << s.eigenvectors() << std::endl;

    int First_ev;
    for(int i = 0; i <n; ++i){
        if(i==0){
            First_ev=i;
        }
        else{
            if(abs(1-s.eigenvalues()[i].real())<abs(1-s.eigenvalues()[First_ev].real()))First_ev=i;
        }
    }
    // Replace 'or' with '||' for C++
    if (abs(1-s.eigenvalues()[First_ev].real())>0.005 || abs(s.eigenvalues()[First_ev].imag())>0.005){
        cout<<s.eigenvalues()[First_ev]<<endl;
        cout<<"ERROR: Matrix is not normalized properly"<<endl;
        abort();
    }
    if(mode=="print_y"){
        cout<<"First eigenvalue (Re,Im) = "<<s.eigenvalues()[First_ev]<<std::endl;
    }
    double sum=0;
    for(int i=0; i<n; i++){
        Pi[i] =s.eigenvectors().col(First_ev)[i].real();
        sum+=Pi[i];
    }
    for(int i=0; i<n; i++){
        Pi[i]*=1.0/sum;
        if(mode=="print_y"){
            cout<<"Pi@"<<i<<" = "<<Pi[i]<<endl;
        }
    }
}



void check_DB(int n, vector<double>& Pi,Eigen::MatrixXd& A){
    cout<<"DBbroken = ";
    double sum=0;
    for(int i = 0; i < n;i++){
        for(int j = 0; j < n; j++){
            sum+= abs(Pi[i]*A(i,j)-Pi[j]*A(j,i));    
    }}
    cout<<std::setprecision(2)<<sum<<endl;
}


void recover_DB(int n, vector<double>& Pi,Eigen::MatrixXd& A){

    for(int i=0;i<n;i++){
        for(int j=i+1;j<n;j++){  
            A(i,j) = A(j,i)*Pi[j]/Pi[i];
        }
    }

}


void calc_A_ini(int n, int T, int Nlin, vector<vector<vector<double> > >& B, Eigen::MatrixXd& A_ini){

    //Compute the least square estimate and then symmetrize it.
    MatrixXd A_LS(n,n);
    MatrixXd L(n,n);
    MatrixXd M(n,n);
    MatrixXd Minv(n,n);
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
        L(i,j)=0;
        M(i,j)=0;
        }
    }
    for (int i = 0; i < n; i++){
    for (int j = 0; j < n; j++){ 
    for (int l = 0; l < Nlin; l++){
    for (int t = 0; t < T-1; t++){
        L(i,j) +=B[l][t+1][i]*B[l][t][j];
        M(i,j) +=B[l][t][i]*B[l][t][j];
    }}}}  
    Minv=M.inverse();
    A_LS = L*Minv;
    A_ini =A_LS;
    //cout<<A_ini<<endl;
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(A_ini(i,j)<0){
                A_ini(i,j) = -1.0*A_ini(i,j);
            }
        };
    };

    normalize_A(n,A_ini);
    sym_A(n,A_ini);
    //cout<<A_ini<<endl;

    // In case diagonal components are negative, use the uniform matrix.
    int cond=0;
    for(int i=0; i<n; i++){
        if(A_ini(i,i)<0){cond=1;}
    }
    if(cond==1){
    for(int i=0; i<n; i++){for(int j=0; j<n;j++){ A_ini(i,j) =1.0/n;}}   
    }

}


void update_Ne(int n, vector<double>& Ne_old,vector<double>& Ne_new){
    Ne_new=Ne_old;
   
    double min,max;
    min=10;//can be set to 0
    max=100000;

    double dN;
    double dN_scale =10;
    // if(Ne_old.size()>10){
    //     dN_scale =5; 
    // } 
    for (int i=0; i<n; i++){
        dN=dN_scale*norm_dist_h(gen_h);
        if (Ne_old[i]+ dN<min){
            Ne_new[i] = 2*min- (Ne_old[i]+ dN);
        }else if (Ne_old[i]+ dN>max){
            Ne_new[i] = 2*max- (Ne_old[i]+ dN);
        }else{
            Ne_new[i]+=dN;
        }
    }
    
}

void update_C(int n, vector<double>& C_old,vector<double>& C_new){
    C_new=C_old;
   
    double min,max;
    min=1; 
    max=10;

    double dC;
    double dC_scale =0.01;

    for (int i=0; i<n; i++){
        dC=dC_scale*norm_dist_h(gen_h);
        if (C_old[i]+ dC<min){
            C_new[i] = 2*min- (C_old[i]+ dC);
        }else if (C_old[i]+ dC>max){
            C_new[i] = 2*max- (C_old[i]+ dC);
        }else{
            C_new[i]+=dC;
        }
    }
    
}


// void update_Ne(int n, vector<double>& Ne_old,vector<double>& Ne_new){
//     Ne_new=Ne_old;
   
//     double min,max;
//     min=4;
//     max=12;

//     double aux;
//     for (int i=0; i<n; i++){
//         Ne_new[i] = log(Ne_old[i])+0.05*norm_dist_h(gen_h);
//         if (Ne_new[i]<min){
//             Ne_new[i] = min + abs(min-Ne_new[i]);
//         }
//         if (Ne_new[i]>max){
//             Ne_new[i] = max - abs(max-Ne_new[i]);
//         }
//     }

//     for (int i=0; i<n; i++){
//         Ne_new[i] = exp(Ne_new[i]);
//     }
    
// }

////////////////////////////////////////////////////////////////
// void pre_update_A(int n, Eigen::MatrixXd& A_old,Eigen::MatrixXd& A_new){
//     double eps=0.0025;
//     A_new=A_old;
//     double da;
//     vector<int> labels_for_A; // Used for updating A
//     for(int i=0;i<n; i++)labels_for_A.push_back(i);
//     for (int i=0; i<n; i++) {
//             shuffle(labels_for_A.begin(), labels_for_A.end(), gen_h);
//             int pos1 = labels_for_A[0];
//             int pos2 = labels_for_A[1];
//             double L = A_old(i,pos1) +  A_old(i,pos2);
//             double r = A_old(i,pos1)/L;
//             double dr = eps*norm_dist_h(gen_h);

//             if(r + dr>1){
//                 r = 2 - r - dr;
//             }else if (r+dr<0)
//             {
//                 r = -r -dr;
//             }else{
//                 r += dr;
//             }
//             A_new(i,pos1) = r*L;
//             A_new(i,pos2) = (1-r)*L;
//         }

// }

// void update_Ne(int n, vector<double>& Ne_old,vector<double>& Ne_new){
//     Ne_new=Ne_old;
//     if(Nupdatemode==0){
//         double min,max;
//         min=10;
//         max=100000;

//         double dN;
//         double dN_scale =10;
//         if(Ne_old.size()>10){
//            dN_scale =5; 
//         } 
//         for (int i=0; i<n; i++){
//             dN=dN_scale*norm_dist(gen);
//             if (Ne_old[i]+ dN<min){
//                 Ne_new[i] = 2*min- (Ne_old[i]+ dN);
//             }else if (Ne_old[i]+ dN>max){
//                 Ne_new[i] = 2*max- (Ne_old[i]+ dN);
//             }else{
//                 Ne_new[i]+=dN;
//             }
//         }
    
//     }else if (Nupdatemode==1){
        
//     for (int i=0; i<Ndeme; i++){Ne_new[i] = Ne_old[i]*exp(5.0*eps*norm_dist(gen));}
    
//     }else if (Nupdatemode==2){
//         double min,max;
//         min=0.00001;
//         max=0.1;
        
//         for (int i=0; i<n; i++) {
//             double Ninv_old=1.0/Ne_old[i];
//             double Ninv_new;
//             double dNinv;
//             dNinv=0.00005*norm_dist(gen);
//             if(Ninv_old+dNinv<min){
//                 Ninv_new = 2*min- (Ninv_old+ dNinv);
//             }else if (Ninv_old+dNinv>max){
//                 Ninv_new  = 2*max- (Ninv_old+ dNinv);
//             }else{
//                 Ninv_new=Ninv_old+dNinv;
//             }
//             Ne_new[i]=1.0/Ninv_new;
//         }

//     }
    
// }
