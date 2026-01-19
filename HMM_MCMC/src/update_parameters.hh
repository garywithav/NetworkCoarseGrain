// Ensure utility functions are available
#include "functions.hh"
using namespace std;
using namespace Eigen;

double DB_reversible_update(int n, vector<double>& Pi_old,vector<double>& Pi_new,Eigen::MatrixXd& A_old,Eigen::MatrixXd& A_new){

    Pi_new = Pi_old;
    int I = randint(0,n-1);
    int J = randint_except(0,n-1,I);

    double Delta_low, Delta_high;
    double aux1,aux2;
    aux1 = A_old(I,J);
    aux2 = Pi_old[J]*A_old(J,I)/Pi_old[I];
    if(aux1>aux2){Delta_high =aux2;}else{Delta_high =aux1;}

    aux1 = -1.0*A_old(I,I);
    aux2 = -1.0*A_old(J,J)* Pi_old[J]/Pi_old[I];
    if(aux1>aux2){Delta_low =aux1;}else{Delta_low =aux2;}
    Delta_high = A_old(I,J);

    double Delta;
    Delta = frand(Delta_low,Delta_high);

    A_new = A_old;
    A_new(I,J) -=Delta;
    A_new(I,I) +=Delta;
    A_new(J,I) -= Pi_old[I]/Pi_old[J]*Delta;
    A_new(J,J) += Pi_old[I]/Pi_old[J]*Delta; 
    
    if (A_new(I,J)<0 || A_new(I,I)<0 || A_new(J,I)<0 || A_new(J,J)<0){
        cout<<A_old;
        cout<<"I,J "<<I<<" "<<J<<endl;
        cout<<"Delta"<<Delta<<" in ["<<Delta_low<<","<<Delta_high<<"]"<<endl;
        cout<<A_new;
        check_A_nonneg(n, A_new);
        check_DB(n,Pi_new, A_new);
    }

    double num,denom;
    num =(A_old(I,J) - Delta)*(A_old(I,J) - Delta) + (A_old(J,I) - Pi_old[I]/Pi_old[J] *Delta)* (A_old(J,I) - Pi_old[I]/Pi_old[J] *Delta);
    denom = A_old(I,J)*A_old(I,J) + A_old(J,I)* A_old(J,I);

    return sqrt(num/denom);
}


double DB_row_update(int n, vector<double>& Pi_old,vector<double>& Pi_new,Eigen::MatrixXd& A_old,Eigen::MatrixXd& A_new){

    int I = randint(0,n-1);
    //cout<<I<<endl;
    double alpha_low, alpha_high;

    alpha_low =0;
    alpha_high =1.0/(1.0-A_old(I,I));
    if(alpha_high<alpha_low){alpha_high=alpha_low;}
    double alpha;
    alpha= frand(alpha_low,alpha_high);
    A_new = A_old;
    for(int j=0;j<n;j++){
        if(j!=I){A_new(I,j) =alpha*A_old(I,j);}
        else{A_new(I,I) = alpha*A_old(I,I)- alpha+1;}
    }

    Pi_new =Pi_old;
    for(int j=0;j<n;j++){
        if(j!=I){
            Pi_new[j] = alpha*Pi_old[j]/(Pi_old[I] + alpha*(1-Pi_old[I]));
            }
        else{
             Pi_new[I] = Pi_old[I]/(Pi_old[I] + alpha*(1-Pi_old[I]));
        }
    }

    return pow(alpha,n-2);
}


double nonrev_update(int n,Eigen::MatrixXd& A_old,Eigen::MatrixXd& A_new){

    int I = randint(0,n-1);
    int J = randint_except(0,n-1,I);

    double Delta_low, Delta_high;
    Delta_low =-1.0*A_old(I,I);
    Delta_high = A_old(I,J);
    double Delta;
    Delta = frand(Delta_low,Delta_high);

    A_new = A_old;
    A_new(I,J) -=Delta;
    A_new(I,I) +=Delta;
    return 1;
}
