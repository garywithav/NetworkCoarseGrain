// --- Windows/MSVC compatibility patch ---
// (All code above this line has been removed. Only the Windows-friendly code remains.)
// --- Windows/MSVC compatibility patch ---
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <sstream>
#include <string>
#include <array>
#include <vector>
#include <random>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <direct.h> // For _mkdir on Windows
#include "src/Eigen/Core"
#include "src/Eigen/Dense"
#include "src/update_parameters.hh"
#include "src/kalman.hh"

using namespace std;
using namespace Eigen;

int Nlin; //Number of (super)lineages
int Ndeme; //Number of demes
int T; //Number of timepoints
int mcmc_max=10000;//Total MCMC steps
double frac_burnin=0.0;//Fraction of burn in
int numprint=10000;//Number of MCMC steps at which the state is recorded
int noise_mode=0;//noise ansatz
int C_mode=0;//0: infer deviation from uniform sampling
double Neff_ini=1000;//Initial population size
double C_ini = 1;//Initial value of deviation from uniform sampling
string Q_DB="nonDB";

int main(int argc, char * argv[])
{
    time_t start_time, end_time;
    start_time = time(NULL);

    string infilename;
    string outfilename;

    srand((unsigned int)time(NULL));

    // Manual argument parsing (replace getopt)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) mcmc_max = atoi(argv[++i]);
        else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) infilename = argv[++i];
        else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) frac_burnin = atof(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) noise_mode = atoi(argv[++i]);
        else if (strcmp(argv[i], "-N") == 0 && i + 1 < argc) Neff_ini = atof(argv[++i]);
        else if (strcmp(argv[i], "-D") == 0 && i + 1 < argc) Q_DB = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) numprint = atoi(argv[++i]);
        else if (strcmp(argv[i], "-C") == 0 && i + 1 < argc) C_mode = atoi(argv[++i]);
    }
    outfilename = infilename;

    _mkdir("output");

    //open output files
    string filename;
    ofstream f_A;
    filename="output/A_"+outfilename+".csv";
    f_A.open(filename);
    ofstream f_Ne;
    filename="output/Ne_"+outfilename+".csv";
    f_Ne.open(filename);
    ofstream f_C;
    filename="output/C_"+outfilename+".csv";
    f_C.open(filename);
    ofstream f_logLH;
    filename="output/logLH_"+outfilename+".csv";
    f_logLH.open(filename);
    ofstream f_log;
    filename="output/logfile_"+outfilename+".csv";
    f_log.open(filename);

    //read input files
    vector<vector<double> >  shape;
    filename="input/shape_"+infilename+".csv";
    Getmatrix(filename, shape);
    T=static_cast<int>(shape[0][0]);
    Nlin=static_cast<int>(shape[1][0]);
    Ndeme=static_cast<int>(shape[2][0]);

    cout<<"T="<<T<<", Nlin="<<Nlin<<", Ndeme="<<Ndeme<<endl<<"noisemode="<<noise_mode<<endl<<Q_DB<<endl;
    f_log<<"T="<<T<<", Nlin="<<Nlin<<", Ndeme="<<Ndeme<<endl<<"noisemode="<<noise_mode<<endl<<Q_DB<<endl;

    filename="input/counts_"+infilename+".csv";
    vector<vector<double> > countsaux;
    Getmatrix(filename, countsaux);
    vector<vector<vector<double> > > counts(Nlin, vector<vector<double> >(T, vector<double>(Ndeme)));
    for (size_t i=0; i<countsaux.size(); i++) {
        int lin_label = static_cast<int>(i/T);
        int t = static_cast<int>(i%T);
        for(int d=0;d<Ndeme;d++){
            counts[lin_label][t][d] =countsaux[i][d];
        }
    }

    filename="input/totcounts_"+infilename+".csv";
    vector<vector<double> > totcounts_aux;
    Getmatrix(filename, totcounts_aux);
    vector<vector<double> > totcounts( Ndeme , vector<double> (T));
    for (int t=0; t<T;t++) {
        for(int i=0;i<Ndeme;i++){
            totcounts[i][t] = totcounts_aux[t][i]+1;
        }
    }

    // Initialize state
    MatrixXd Imat = MatrixXd::Identity(Ndeme, Ndeme);
    MatrixXd A_ini(Ndeme,Ndeme);
    calc_A_ini(Ndeme, T, Nlin, counts, A_ini);
    MatrixXd A_old = A_ini;
    vector<double> Ne_old (Ndeme, Neff_ini);
    vector<double> C_old (Ndeme, C_ini);
    vector<vector<double> > totcounts_eff( Ndeme , vector<double> (T));
    for (int t=0; t<T;t++) {
        for(int i=0;i<Ndeme;i++){
            totcounts_eff[i][t] = totcounts[i][t]/C_old[i];
        }
    }
    vector<double> Pi_old (Ndeme);
    calc_Pi(Ndeme, A_old,  Pi_old,"print_n");
    double logLH_old;
    cout<<"Aold"<<endl<<A_old<<endl;
    logLH_old = calc_LH(Nlin, Ndeme, T, counts, Ne_old, totcounts_eff, A_old, noise_mode, Imat);
    cout<<"logLH_old"<< std::fixed << std::setprecision(8)<<logLH_old<<endl;
    int accept=0;
    //Declare variables used for proposed state
    MatrixXd A_new(Ndeme,Ndeme);
    vector<double> Pi_new (Ndeme);
    vector<double> Ne_new(Ndeme);
    vector<double> C_new(Ndeme);
    int dstep=(int)(mcmc_max*(1-frac_burnin)/(double)numprint);
    if (dstep<1) { dstep=1;}
    int dstep_checkpoint=(int)((double)mcmc_max/20.0);
    if(dstep_checkpoint<1){dstep_checkpoint=1;}
    for (int mcmc_step=0; mcmc_step<mcmc_max; mcmc_step++) {
        if(mcmc_step%dstep_checkpoint==0 &&mcmc_step>0){
            calc_Pi(Ndeme, A_old,  Pi_old,"print_n");
            cout<<"Step = "<<mcmc_step;
            cout<<", "<<" acc = "<<round((double)100.0*accept/dstep_checkpoint)<<"%, ";
            cout<<"lnLH="<<std::fixed << std::setprecision(6)<<logLH_old<<", ";
            check_DB(Ndeme, Pi_old,A_old);
            f_log<<mcmc_step<<", "<<" acc (%)= "<<round((double)100.0*accept/dstep_checkpoint)<<", ";
            f_log<<"lnLH="<<logLH_old<<endl;
            accept=0;
        }
        double p_proposal;
        double logLH_new;
        double p_acc;
        if(Q_DB=="DB"){
            normalize_A(Ndeme,A_old);
            calc_Pi(Ndeme, A_old,  Pi_old,"print_n");
            recover_DB(Ndeme, Pi_old,A_old);
            double r1 = ((double)rand() / RAND_MAX);
            if(r1<0.5){
                update_Ne(Ndeme, Ne_old,Ne_new);
                if(C_mode==0){
                    update_C(Ndeme, C_old,C_new);
                }
                else{
                    C_new = C_old;
                }
                p_proposal=DB_reversible_update(Ndeme,Pi_old,Pi_new, A_old,A_new);
            }
            else{
                update_Ne(Ndeme, Ne_old,Ne_new);
                if(C_mode==0){
                    update_C(Ndeme, C_old,C_new);
                }
                else{
                    C_new = C_old;
                }
                p_proposal=DB_row_update(Ndeme,Pi_old,Pi_new, A_old,A_new);
            }
        }
        else if(Q_DB=="nonDB"){
            normalize_A(Ndeme,A_old);
            update_Ne(Ndeme, Ne_old,Ne_new);
            if(C_mode==0){
                update_C(Ndeme, C_old,C_new);
            }
            else{
                C_new = C_old;
            }
            p_proposal=nonrev_update(Ndeme, A_old,A_new);
        }
        else{
            cout<<"Specify DB or nonDB"<<endl;
            return 0;
        }
        for (int t=0; t<T;t++) {
            for(int i=0;i<Ndeme;i++){
                totcounts_eff[i][t] = totcounts[i][t]/C_new[i];
            }
        }
    logLH_new = calc_LH(Nlin, Ndeme, T, counts, Ne_new, totcounts_eff, A_new, noise_mode, Imat);
        p_acc =  p_proposal*exp(logLH_new-logLH_old);
        double r = ((double)rand() / RAND_MAX);
        if(r<p_acc){
            logLH_old=logLH_new;
            A_old =A_new;
            Ne_old=Ne_new;
            C_old=C_new;
            Pi_old = Pi_new;
            accept+=1;
        }
        if(mcmc_step%dstep==0 && mcmc_step>=frac_burnin*mcmc_max) {
            for (int i=0; i<Ndeme; i++) {
                for (int j=0; j<Ndeme;j++) {
                    if (i==Ndeme-1 && j==Ndeme-1) {
                        f_A<<A_old(i,j)<<endl;
                    }else{
                        f_A<<A_old(i,j)<<",";
                    }
                }
            }
            for (int i=0; i<Ndeme; i++) {
                if(i<Ndeme-1){
                    f_Ne<<Ne_old[i]<<",";
                }else{
                    f_Ne<<Ne_old[i]<<endl;
                }
            }
            for (int i=0; i<Ndeme; i++) {
                if(i<Ndeme-1){
                    f_C<<C_old[i]<<",";
                }else{
                    f_C<<C_old[i]<<endl;
                }
            }
            f_logLH<<mcmc_step <<","<<logLH_old<<endl;
        }
    }
    end_time = time(NULL);
    cout<<"Afinal"<<endl<<A_old<<endl;
    f_log << "Run time: " << double(end_time-start_time)<< endl;
    return 0;
}
