#include "fourstep.hpp"
#include <sys/resource.h>
#include <chrono>
#include <cstdio>
#include <cstring>
using namespace kfr;
static double rss(){ struct rusage r; getrusage(RUSAGE_SELF,&r); return r.ru_maxrss/1048576.0; }
static double now(){ return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
template<class T> static void fill(complex<T>* a, size_t n){ for(size_t i=0;i<n;i++) a[i]=complex<T>(T(std::sin(0.001*i)+((i*2654435761u)>>28)*0.01), T(std::cos(0.37*i))); }
int main(int argc,char**argv){
    size_t n=strtoull(argv[1],0,10); const char* mode=argv[2];
    if(!strcmp(mode,"accuracy")){ // f32 four-step and f32 direct vs f64 direct
        univector<complex<double>> ref(n); fill(ref.data(),n);
        univector<complex<float>> a(n), o1(n), o2(n);
        for(size_t i=0;i<n;i++) a[i]=complex<float>(ref[i]);
        { dft_plan<double> p(n); univector<u8> t(p.temp_size); p.execute(ref.data(),ref.data(),t.data()); }
        { dft_plan<float> p(n); univector<u8> t(p.temp_size); p.execute(o1.data(),a.data(),t.data()); }
        { fft_fourstep<float> f(n); printf("N1=%zu N2=%zu\n",f.n1(),f.n2()); f.execute(o2.data(),a.data(),false); }
        double e1=0,e2=0,s=0; for(size_t i=0;i<n;i++){ s+=std::norm(ref[i]); e1+=std::norm(complex<double>(o1[i])-ref[i]); e2+=std::norm(complex<double>(o2[i])-ref[i]); }
        printf("n=%zu rel_rms kfr_direct_f32=%.3g fourstep_f32=%.3g\n",n,sqrt(e1/s),sqrt(e2/s));
        // inverse round trip of four-step
        fft_fourstep<float> f(n); univector<complex<float>> back(n); f.execute(back.data(),o2.data(),true);
        univector<complex<float>> a0(n); fill(a0.data(),n); double e3=0,s3=0; for(size_t i=0;i<n;i++){ s3+=std::norm(complex<double>(a0[i])); e3+=std::norm(complex<double>(back[i])/double(n)-complex<double>(a0[i])); }
        printf("  fourstep inverse round-trip rel_rms=%.3g\n",sqrt(e3/s3));
        return 0;
    }
    bool f64=argc>3 && !strcmp(argv[3],"f64");
    auto go=[&](auto tag){ using T=decltype(tag);
        univector<complex<T>> in(n), out(n); fill(in.data(),n);
        double t0=now();
        if(!strcmp(mode,"direct")){ dft_plan<T> p(n); univector<u8> t(p.temp_size); double t1=now();
            p.execute(out.data(),in.data(),t.data()); double t2=now();
            printf("direct   %s n=%zu plan=%.2fs exec=%.2fs peak_rss=%.3fGiB\n", sizeof(T)==4?"f32":"f64", n,t1-t0,t2-t1,rss()); }
        else { fft_fourstep<T> f(n); double t1=now(); f.execute(out.data(),in.data(),false); double t2=now();
            printf("fourstep %s n=%zu (N1=%zu N2=%zu, extra=%.1fMiB) plan=%.2fs exec=%.2fs peak_rss=%.3fGiB\n", sizeof(T)==4?"f32":"f64", n,f.n1(),f.n2(),f.extra_bytes()/1048576.0,t1-t0,t2-t1,rss()); }
    };
    if(f64) go(double()); else go(float());
}
