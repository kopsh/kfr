#include <kfr/dft.hpp>
#include <sys/resource.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace kfr;
static double rss(){ struct rusage r; getrusage(RUSAGE_SELF,&r); return r.ru_maxrss/1048576.0; }
template<class T> void run(size_t n, bool real, bool exec, bool inplace){
    double g=1024.0*1024*1024;
    if(!real){
        dft_plan<T> p(n);
        printf("complex<%s> n=%zu data=%.3fG temp=%.3fG stages=%zu  rss_after_plan=%.3fG\n", sizeof(T)==4?"f32":"f64", n,
               p.data_size/g, p.temp_size/g, p.all_stages.size(), rss());
        if(getenv("DUMP")) p.dump();
        if(exec){
            univector<complex<T>> a(n), b(inplace?0:n); univector<u8> t(p.temp_size);
            for(size_t i=0;i<n;i++) a[i]=complex<T>(T(i%7),T(1));
            p.execute(inplace?a.data():b.data(), a.data(), t.data());
            printf("  exec %s: peak rss=%.3fG\n", inplace?"inplace":"out-of-place", rss());
        }
    } else {
        dft_plan_real<T> p(n);
        printf("real<%s> n=%zu data=%.3fG temp=%.3fG rss_after_plan=%.3fG\n", sizeof(T)==4?"f32":"f64", n, p.data_size/g, p.temp_size/g, rss());
        if(exec){
            univector<T> a(n); univector<complex<T>> b(n/2+1); univector<u8> t(p.temp_size);
            for(size_t i=0;i<n;i++) a[i]=T(i%7);
            p.execute(b.data(), a.data(), t.data());
            printf("  exec: peak rss=%.3fG\n", rss());
        }
    }
}
int main(int c,char**v){
    size_t n=strtoull(v[1],0,10); bool f64=!strcmp(v[2],"f64"); bool real=!strcmp(v[3],"real");
    bool exec=c>4 && strcmp(v[4],"noexec"); bool inplace=c>4 && !strcmp(v[4],"inplace");
    if(f64) run<double>(n,real,exec,inplace); else run<float>(n,real,exec,inplace);
}
