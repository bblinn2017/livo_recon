// Both AdaptiveQ channels vs n_cp, on a MOVING-axis rotation truth with
// injected white IMU noise, poses dead-reckoned from that same noisy IMU.
#include "livo_recon/lio/spline.h"
#include <cstdio>
#include <vector>
#include <random>
using namespace livo_recon;
struct Truth {
  double A,B,c1,c2,w1,w2;
  double alpha(double t)const{return A*std::sin(w1*t)+c1*t;}
  double alphadot(double t)const{return A*w1*std::cos(w1*t)+c1;}
  double beta(double t)const{return B*std::cos(w2*t)+c2*t;}
  double betadot(double t)const{return -B*w2*std::sin(w2*t)+c2;}
  M3D rot(double t)const{return Exp(V3D(0,0,1)*alpha(t))*Exp(V3D(1,0,0)*beta(t));}
  V3D omega(double t)const{return Exp(V3D(1,0,0)*beta(t)).transpose()*V3D(0,0,1)*alphadot(t)+V3D(1,0,0)*betadot(t);}
  V3D accW(double t)const{return V3D(-0.9*2.1*2.1*std::sin(2.1*t),-0.6*1.7*1.7*std::cos(1.7*t),-0.25*3.3*3.3*std::sin(3.3*t));}
  V3D posW(double t)const{return V3D(0.9*std::sin(2.1*t)+0.3*t,0.6*std::cos(1.7*t),0.25*std::sin(3.3*t));}
  V3D velW(double t)const{return V3D(0.9*2.1*std::cos(2.1*t)+0.3,-0.6*1.7*std::sin(1.7*t),0.25*3.3*std::cos(3.3*t));}
};
int main(){
  const double t0=0,t1=0.1; const V3D g(0,0,-9.81);
  const double sa=0.02, sg=0.002;          // true injected sigmas
  const int N=20; const double dt=(t1-t0)/N;
  printf("Both channels vs n_cp -- MOVING-axis rotation, injected sigma_a=%.4f sigma_g=%.5f\n",sa,sg);
  printf("Allan gyro floor from the bags: 1.65e-4 to 7.39e-4 rad/s\n\n");
  for(double sc : {0.25, 1.0}){
    Truth T{0.35*sc,0.28*sc,0.8*sc,0.6*sc,24.0,31.0};
    double chord=Log(T.rot(t0).transpose()*T.rot(t1)).norm()*180/M_PI;
    std::mt19937 rng(4242); std::normal_distribution<double> na(0,sa),ng(0,sg);
    std::vector<ImuSample> imu;
    for(int i=0;i<=N;++i){double t=t0+i*dt;M3D R=T.rot(t);
      imu.emplace_back(R.transpose()*(T.accW(t)-g)+V3D(na(rng),na(rng),na(rng)),
                       T.omega(t)+V3D(ng(rng),ng(rng),ng(rng)),t);}
    std::vector<Pose6D> poses; M3D R=T.rot(t0); V3D p=T.posW(t0), v=T.velW(t0);
    for(int i=0;i<N;++i){Pose6D ps{};ps.t=imu[i].t;ps.rot=R;ps.pos=p;ps.vel=v;ps.dt=dt;poses.push_back(ps);
      const V3D a0=R*imu[i].acc+g; R=R*Exp(0.5*(imu[i].gyro+imu[i+1].gyro),dt);
      const V3D a1=R*imu[i+1].acc+g; const V3D aw=0.5*(a0+a1);
      p+=v*dt+0.5*aw*dt*dt; v+=aw*dt;}
    printf("chord %.2f deg\n  n_cp  sigma_a_hat  ratio   acf1_a   sigma_g_hat   ratio   acf1_g\n",chord);
    for(int ncp : {4,6,8,10,12,16,20,24,32}){
      SplineOptions o;o.n_control_points=ncp;o.rot_mode="tangent";
      ScanSpline s; if(!s.fit(poses,poses.front().t,t1,o)){printf("  %4d  fit refused\n",ncp);continue;}
      auto st=computeSplineImuResidual(s,imu,V3D::Zero(),V3D::Zero(),g);
      if(!st.valid()){printf("  %4d  n too small\n",ncp);continue;}
      printf("  %4d  %10.6f  %6.2f  %+7.3f  %11.7f  %6.2f  %+7.3f\n",s.nControlPoints(),
        std::sqrt(st.cov_acc),std::sqrt(st.cov_acc)/sa,st.acf1_acc,
        std::sqrt(st.cov_gyr),std::sqrt(st.cov_gyr)/sg,st.acf1_gyr);
    }
    printf("\n");
  }
  return 0;
}
