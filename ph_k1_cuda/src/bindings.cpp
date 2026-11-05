// Python module ph_k1_native: the constants block from PyTorch tensors, and `run`, which advances a
// batch of robots in place (CPU or CUDA, float32 or float64, decided by the tensors).
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cstring>
#include <map>
#include <string>

#ifdef PH_K1_WITH_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif

#include "ph_k1/launch.hpp"

namespace {

using TensorMap = std::map<std::string, torch::Tensor>;

torch::Tensor field(const TensorMap& t, const std::string& name, int64_t numel) {
    auto it = t.find(name);
    TORCH_CHECK(it != t.end(), "pack_model: missing tensor '", name, "'");
    auto x = it->second.detach().to(torch::kCPU, torch::kFloat64).contiguous();
    TORCH_CHECK(x.numel() == numel, "pack_model: '", name, "' has ", x.numel(), " values, expected ", numel);
    return x;
}

template <class T>
void put(T* dst, const TensorMap& t, const std::string& name, int64_t numel) {
    auto x = field(t, name, numel);
    const double* s = x.data_ptr<double>();
    for (int64_t i = 0; i < numel; ++i) dst[i] = static_cast<T>(s[i]);
}

void put_int(int* dst, const TensorMap& t, const std::string& name, int64_t numel) {
    auto x = field(t, name, numel);
    const double* s = x.data_ptr<double>();
    for (int64_t i = 0; i < numel; ++i) dst[i] = static_cast<int>(s[i]);
}

template <class T>
torch::Tensor pack(const TensorMap& t, const std::map<std::string, double>& s, const std::map<std::string, int64_t>& n) {
    auto num = [&](const std::string& k) {
        auto it = s.find(k);
        TORCH_CHECK(it != s.end(), "pack_model: missing scalar '", k, "'");
        return it->second;
    };
    auto flag = [&](const std::string& k) {
        auto it = n.find(k);
        TORCH_CHECK(it != n.end(), "pack_model: missing int '", k, "'");
        return static_cast<int>(it->second);
    };
    ph::ModelData<T> m;
    std::memset(&m, 0, sizeof(m));
    m.nbody = flag("nbody");
    m.ncontact = flag("ncontact");
    TORCH_CHECK(m.nbody > 0 && m.nbody <= ph::MAX_BODIES, "nbody ", m.nbody, " outside 1..", ph::MAX_BODIES);
    TORCH_CHECK(m.ncontact >= 0 && m.ncontact <= ph::MAX_CONTACTS, "ncontact ", m.ncontact, " outside 0..", ph::MAX_CONTACTS);
    m.fixed_base = flag("fixed_base");
    m.motors_connected = flag("motors_connected");
    m.gravity = flag("gravity");
    m.contact_enabled = flag("contact_enabled");
    m.limits_enabled = flag("limits_enabled");
    m.h = static_cast<T>(num("h"));
    put(m.g, t, "g", 3);
    const int nb = m.nbody, nc = m.ncontact;
    put_int(m.parent, t, "parent", nb);
    put_int(m.body_joint, t, "body_joint", nb);
    put_int(m.jnt_body, t, "jnt_body", ph::NJ);
    put_int(m.jnt_parent_body, t, "jnt_parent_body", ph::NJ);
    put(&m.body_pos[0][0], t, "body_pos", nb * 3);
    put(&m.body_rot[0][0], t, "body_rot", nb * 9);
    put(&m.jnt_axis[0][0], t, "jnt_axis", ph::NJ * 3);
    put(&m.jnt_pos[0][0], t, "jnt_pos", ph::NJ * 3);
    put(m.mass, t, "mass", nb);
    put(&m.ipos[0][0], t, "ipos", nb * 3);
    put(&m.I_body[0][0], t, "I_body", nb * 9);
    put(m.armature, t, "armature", ph::NJ);
    put(m.jnt_low, t, "jnt_low", ph::NJ);
    put(m.jnt_high, t, "jnt_high", ph::NJ);
    put(m.stop_low, t, "stop_low", ph::NJ);
    put(m.stop_high, t, "stop_high", ph::NJ);
    m.lim_k = static_cast<T>(num("lim_k"));
    m.lim_c = static_cast<T>(num("lim_c"));
    if (nc > 0) {
        put_int(m.contact_body, t, "contact_body", nc);
        put(&m.contact_local[0][0], t, "contact_local", nc * 3);
    }
    m.k_n = static_cast<T>(num("k_n"));
    m.c_n = static_cast<T>(num("c_n"));
    m.k_t = static_cast<T>(num("k_t"));
    m.c_t = static_cast<T>(num("c_t"));
    m.mu = static_cast<T>(num("mu"));
    put(m.mk, t, "motor_k", ph::NJ);
    put(m.mR, t, "motor_R", ph::NJ);
    put(m.mL, t, "motor_L", ph::NJ);
    put(m.tau_max, t, "motor_tau_max", ph::NJ);
    put(m.damping, t, "motor_damping", ph::NJ);
    m.v_max = static_cast<T>(num("v_max"));
    m.fw_pos_clamp = flag("fw_pos_clamp");
    m.fw_slew = flag("fw_slew");
    m.fw_target_clamp = flag("fw_target_clamp");
    m.fw_overspeed = flag("fw_overspeed");
    m.policy_dt = static_cast<T>(num("policy_dt"));
    put(m.fw_effort, t, "fw_effort", ph::NJ);
    put(m.fw_max_joint_velocity, t, "fw_max_joint_velocity", ph::NJ);
    put(m.fw_max_target_velocity, t, "fw_max_target_velocity", ph::NJ);
    for (int k = 0; k < ph::NJ; ++k)
        TORCH_CHECK(m.stop_low[k] <= m.jnt_low[k] && m.stop_high[k] >= m.jnt_high[k],
                    "end stop of hinge ", k, " lies inside its command range");
    TORCH_CHECK(m.body_joint[0] == -2 && m.parent[0] == -1, "body 0 must be the free trunk");
    for (int b = 1; b < nb; ++b) TORCH_CHECK(m.parent[b] >= 0 && m.parent[b] < b, "bodies must be in tree order");

    auto out = torch::empty({static_cast<int64_t>(sizeof(m))}, torch::dtype(torch::kUInt8));
    std::memcpy(out.data_ptr<uint8_t>(), &m, sizeof(m));
    return out;
}

torch::Tensor pack_model(const TensorMap& tensors, const std::map<std::string, double>& scalars,
                         const std::map<std::string, int64_t>& ints, bool use_double) {
    return use_double ? pack<double>(tensors, scalars, ints) : pack<float>(tensors, scalars, ints);
}

template <class T>
T* rows(const c10::optional<torch::Tensor>& x, const torch::Tensor& ref, int64_t B, int64_t width, const char* name,
        bool required = false) {
    if (!x.has_value() || !x->defined()) {
        TORCH_CHECK(!required, name, " is required");
        return nullptr;
    }
    const auto& t = *x;
    TORCH_CHECK(t.device() == ref.device(), name, " is on ", t.device(), ", the state on ", ref.device());
    TORCH_CHECK(t.scalar_type() == ref.scalar_type(), name, " must have dtype ", ref.scalar_type());
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.numel() == B * width, name, " must have ", B, " x ", width, " values, has ", t.numel());
    return t.data_ptr<T>();
}

int* rows_i32(const c10::optional<torch::Tensor>& x, const torch::Tensor& ref, int64_t B, const char* name) {
    TORCH_CHECK(x.has_value() && x->defined(), name, " is required in policy-step mode");
    const auto& t = *x;
    TORCH_CHECK(t.device() == ref.device(), name, " is on the wrong device");
    TORCH_CHECK(t.scalar_type() == torch::kInt32 && t.is_contiguous() && t.numel() == B,
                name, " must be a contiguous int32 tensor of ", B);
    return t.data_ptr<int>();
}

uint8_t* rows_u8(const c10::optional<torch::Tensor>& x, const torch::Tensor& ref, int64_t B, int64_t width, const char* name) {
    if (!x.has_value() || !x->defined()) return nullptr;
    const auto& t = *x;
    TORCH_CHECK(t.device() == ref.device(), name, " is on the wrong device");
    TORCH_CHECK(t.scalar_type() == torch::kUInt8 && t.is_contiguous() && t.numel() == B * width,
                name, " must be a contiguous uint8 tensor of ", B, " x ", width);
    return t.data_ptr<uint8_t>();
}

template <class T>
void run_t(const torch::Tensor& model, const torch::Tensor& q, const torch::Tensor& p, const torch::Tensor& xi,
           const torch::Tensor& lam, const torch::Tensor& ledger, const c10::optional<torch::Tensor>& volts,
           const c10::optional<torch::Tensor>& jtau, const c10::optional<torch::Tensor>& target,
           const c10::optional<torch::Tensor>& last_target, const c10::optional<torch::Tensor>& ring,
           const c10::optional<torch::Tensor>& ring_idx, const c10::optional<torch::Tensor>& lag,
           const c10::optional<torch::Tensor>& push, const c10::optional<torch::Tensor>& kp, const c10::optional<torch::Tensor>& kd, int64_t nsub,
           const c10::optional<torch::Tensor>& out_nu, const c10::optional<torch::Tensor>& out_i,
           const c10::optional<torch::Tensor>& out_tau, const c10::optional<torch::Tensor>& out_v,
           const c10::optional<torch::Tensor>& out_fc, const c10::optional<torch::Tensor>& out_active,
           const c10::optional<torch::Tensor>& out_slip, const c10::optional<torch::Tensor>& out_gyro,
           const c10::optional<torch::Tensor>& out_acc, const c10::optional<torch::Tensor>& out_target,
           const c10::optional<torch::Tensor>& ring_kp, const c10::optional<torch::Tensor>& ring_kd,
           const c10::optional<torch::Tensor>& trunk_force, const c10::optional<torch::Tensor>& gen_force,
           const c10::optional<torch::Tensor>& robot_params) {
    TORCH_CHECK(model.scalar_type() == torch::kUInt8 && model.numel() == static_cast<int64_t>(sizeof(ph::ModelData<T>)),
                "model block was packed for the other float type (or is corrupt)");
    TORCH_CHECK(model.device() == q.device(), "model block is on ", model.device(), ", the state on ", q.device());
    TORCH_CHECK(q.dim() == 2 && q.size(1) == ph::NQ, "q must be (B, ", ph::NQ, ")");
    const int64_t B = q.size(0);
    // ncontact: the CPU copy is read directly; on the GPU it is passed from the xi shape
    TORCH_CHECK(xi.numel() % (2 * std::max<int64_t>(B, 1)) == 0, "xi must be (B, ncontact, 2)");
    const int64_t nc = B > 0 ? xi.numel() / (2 * B) : 0;

    ph::BatchT<T> b;
    b.B = static_cast<int>(B);
    b.nsub = static_cast<int>(nsub);
    b.q = rows<T>(q, q, B, ph::NQ, "q", true);
    b.p = rows<T>(p, q, B, ph::NV, "p", true);
    b.xi = rows<T>(xi, q, B, nc * 2, "xi", true);
    b.lam = rows<T>(lam, q, B, ph::NJ, "lam", true);
    b.ledger = rows<T>(ledger, q, B, ph::LEDGER, "ledger", true);
    b.volts = rows<T>(volts, q, B, ph::NJ, "volts");
    b.jtau = rows<T>(jtau, q, B, ph::NJ, "joint_torque");
    b.policy_target = rows<T>(target, q, B, ph::NJ, "target");
    if (b.policy_target) {
        TORCH_CHECK(!b.volts && !b.jtau, "give either target (policy step) or volts/joint_torque, not both");
        b.last_target = rows<T>(last_target, q, B, ph::NJ, "last_target", true);
        TORCH_CHECK(ring.has_value() && ring->defined() && B > 0 && ring->numel() % (B * ph::NJ) == 0,
                    "ring must be (B, ring_len, 22)");
        b.ring_len = static_cast<int>(ring->numel() / (B * ph::NJ));
        TORCH_CHECK(b.ring_len >= 1, "ring_len must be >= 1");
        b.ring = rows<T>(ring, q, B, b.ring_len * ph::NJ, "ring", true);
        b.ring_idx = rows_i32(ring_idx, q, B, "ring_idx");
        b.lag = rows_i32(lag, q, B, "lag");
        b.push = rows<T>(push, q, B, 3, "push");
        b.kp = rows<T>(kp, q, B, ph::NJ, "kp", true);
        b.kd = rows<T>(kd, q, B, ph::NJ, "kd", true);
        b.ring_kp = rows<T>(ring_kp, q, B, b.ring_len * ph::NJ, "ring_kp");
        b.ring_kd = rows<T>(ring_kd, q, B, b.ring_len * ph::NJ, "ring_kd");
        TORCH_CHECK((b.ring_kp == nullptr) == (b.ring_kd == nullptr), "give ring_kp and ring_kd together");
    }
    b.force = rows<T>(trunk_force, q, B, 3, "trunk_force");
    b.gen_force = rows<T>(gen_force, q, B, ph::NV, "gen_force");
    b.robot_params = rows<T>(robot_params, q, B, ph::NRP, "robot_params");
    b.out_nu = rows<T>(out_nu, q, B, ph::NV, "out_nu");
    b.out_i = rows<T>(out_i, q, B, ph::NJ, "out_i");
    b.out_tau = rows<T>(out_tau, q, B, ph::NJ, "out_tau");
    b.out_v = rows<T>(out_v, q, B, ph::NJ, "out_v");
    b.out_fc = rows<T>(out_fc, q, B, nc * 3, "out_fc");
    b.out_active = rows_u8(out_active, q, B, nc, "out_active");
    b.out_slip = rows_u8(out_slip, q, B, nc, "out_slip");
    b.out_gyro = rows<T>(out_gyro, q, B, 3, "out_gyro");
    b.out_acc = rows<T>(out_acc, q, B, 3, "out_acc");
    b.out_target = rows<T>(out_target, q, B, ph::NJ, "out_target");

    if (q.is_cuda()) {
#ifdef PH_K1_WITH_CUDA
        const auto* md = reinterpret_cast<const ph::ModelData<T>*>(model.data_ptr<uint8_t>());
        ph::run_cuda<T>(md, static_cast<int>(nc), b, at::cuda::getCurrentCUDAStream().stream());
#else
        TORCH_CHECK(false, "ph_k1_native was built without CUDA (no nvcc at build time)");
#endif
    } else {
        const auto& m = *reinterpret_cast<const ph::ModelData<T>*>(model.data_ptr<uint8_t>());
        TORCH_CHECK(m.ncontact == nc, "xi has ", nc, " contact points, the model ", m.ncontact);
        ph::run_cpu<T>(m, b);
    }
}

void run(const torch::Tensor& model, const torch::Tensor& q, const torch::Tensor& p, const torch::Tensor& xi,
         const torch::Tensor& lam, const torch::Tensor& ledger, const c10::optional<torch::Tensor>& volts,
         const c10::optional<torch::Tensor>& jtau, const c10::optional<torch::Tensor>& target,
         const c10::optional<torch::Tensor>& last_target, const c10::optional<torch::Tensor>& ring,
         const c10::optional<torch::Tensor>& ring_idx, const c10::optional<torch::Tensor>& lag,
         const c10::optional<torch::Tensor>& push, const c10::optional<torch::Tensor>& kp, const c10::optional<torch::Tensor>& kd, int64_t nsub,
         const c10::optional<torch::Tensor>& out_nu, const c10::optional<torch::Tensor>& out_i,
         const c10::optional<torch::Tensor>& out_tau, const c10::optional<torch::Tensor>& out_v,
         const c10::optional<torch::Tensor>& out_fc, const c10::optional<torch::Tensor>& out_active,
         const c10::optional<torch::Tensor>& out_slip, const c10::optional<torch::Tensor>& out_gyro,
         const c10::optional<torch::Tensor>& out_acc, const c10::optional<torch::Tensor>& out_target,
           const c10::optional<torch::Tensor>& ring_kp, const c10::optional<torch::Tensor>& ring_kd,
           const c10::optional<torch::Tensor>& trunk_force, const c10::optional<torch::Tensor>& gen_force,
           const c10::optional<torch::Tensor>& robot_params) {
    TORCH_CHECK(nsub >= 1, "nsub must be >= 1");
    if (q.scalar_type() == torch::kFloat64) {
        run_t<double>(model, q, p, xi, lam, ledger, volts, jtau, target, last_target, ring, ring_idx, lag, push, kp, kd, nsub,
                      out_nu, out_i, out_tau, out_v, out_fc, out_active, out_slip, out_gyro, out_acc, out_target,
                      ring_kp, ring_kd, trunk_force, gen_force, robot_params);
    } else if (q.scalar_type() == torch::kFloat32) {
        run_t<float>(model, q, p, xi, lam, ledger, volts, jtau, target, last_target, ring, ring_idx, lag, push, kp, kd, nsub,
                     out_nu, out_i, out_tau, out_v, out_fc, out_active, out_slip, out_gyro, out_acc, out_target,
                      ring_kp, ring_kd, trunk_force, gen_force, robot_params);
    } else {
        TORCH_CHECK(false, "state must be float32 or float64");
    }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.doc() = "K1 port-Hamiltonian plant in C++/CUDA (see ph_k1_cuda/README.md)";
    mod.def("pack_model", &pack_model, "constants block (uint8 CPU tensor) from the model tensors",
            py::arg("tensors"), py::arg("scalars"), py::arg("ints"), py::arg("use_double"));
    mod.def("run", &run, "advance every robot `nsub` plant steps in place (volts mode, or policy-step mode "
            "through the robot's command path when `target` is given)", py::arg("model"), py::arg("q"),
            py::arg("p"), py::arg("xi"), py::arg("lam"), py::arg("ledger"), py::arg("volts") = py::none(),
            py::arg("joint_torque") = py::none(), py::arg("target") = py::none(), py::arg("last_target") = py::none(),
            py::arg("ring") = py::none(), py::arg("ring_idx") = py::none(), py::arg("lag") = py::none(),
            py::arg("push") = py::none(),
            py::arg("kp") = py::none(), py::arg("kd") = py::none(), py::arg("nsub") = 1,
            py::arg("out_nu") = py::none(), py::arg("out_i") = py::none(), py::arg("out_tau") = py::none(),
            py::arg("out_v") = py::none(), py::arg("out_fc") = py::none(), py::arg("out_active") = py::none(),
            py::arg("out_slip") = py::none(), py::arg("out_gyro") = py::none(), py::arg("out_acc") = py::none(),
            py::arg("out_target") = py::none(), py::arg("ring_kp") = py::none(), py::arg("ring_kd") = py::none(),
            py::arg("trunk_force") = py::none(), py::arg("gen_force") = py::none(),
            py::arg("robot_params") = py::none());
    mod.attr("NQ") = ph::NQ;
    mod.attr("NV") = ph::NV;
    mod.attr("NJ") = ph::NJ;
    mod.attr("LEDGER") = ph::LEDGER;
    mod.attr("NRP") = ph::NRP;
#ifdef PH_K1_WITH_CUDA
    mod.attr("WITH_CUDA") = true;
#else
    mod.attr("WITH_CUDA") = false;
#endif
}
