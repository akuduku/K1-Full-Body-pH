// CPU launcher: one robot per OpenMP iteration, each thread with its own work buffers.
#include <memory>

#include "ph_k1/launch.hpp"

namespace ph {

template <class T>
void run_cpu(const ModelData<T>& m, const BatchT<T>& b) {
#pragma omp parallel
    {
        auto w = std::make_unique<Work<T>>();
#pragma omp for schedule(static)
        for (int r = 0; r < b.B; ++r) {
            const StepIO<T> io = make_io(b, r, m.ncontact);
            for (int s = 0; s < b.nsub; ++s) plant_step(m, io, *w);
        }
    }
}

template void run_cpu<float>(const ModelData<float>&, const BatchT<float>&);
template void run_cpu<double>(const ModelData<double>&, const BatchT<double>&);

}  // namespace ph
