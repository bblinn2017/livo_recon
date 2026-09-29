#include "livo_recon/map/stationary/stationary_map.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

// R53: every check is a CHECK(), which is NOT compiled out by NDEBUG (R52 used
// assert(), and CMakeLists.txt sets CMAKE_BUILD_TYPE Release, so the R52
// assertions may never have executed). Test data is 2-D planar (rank 2); the R52
// merge/robust tests used collinear points, which are rank-1 cells with an
// undefined normal.
#define CHECK(c)                                                                        \
  do {                                                                                  \
    if (!(c)) {                                                                         \
      std::fprintf(stderr, "CHECK FAILED %s:%d: %s\n", __FILE__, __LINE__, #c);         \
      std::exit(1);                                                                     \
    }                                                                                   \
  } while (0)

using namespace livo_recon::stationary_map;

namespace {

std::vector<V3> grid(double x0, double y0, int nx, int ny, double step, const std::function<double(double, double)>& z) {
  std::vector<V3> out;
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j) {
      const double x = x0 + i * step, y = y0 + j * step;
      out.emplace_back(x, y, z(x, y));
    }
  return out;
}

PlaneFit batchFit(const std::vector<V3>& pts) {
  V3 mean = V3::Zero();
  for (const auto& p : pts) mean += p;
  mean /= double(pts.size());
  M3 cov = M3::Zero();
  for (const auto& p : pts) { V3 d = p - mean; cov += d * d.transpose(); }
  cov /= double(pts.size());
  Eigen::SelfAdjointEigenSolver<M3> es(cov);
  PlaneFit f;
  f.eigenvalues = es.eigenvalues();
  f.center = mean;
  f.normal = es.eigenvectors().col(0).normalized();
  return f;
}

void checkStats() {
  auto pts = grid(0.0, 0.0, 6, 5, 0.03, [](double x, double y) { return 1.5 + 0.1 * x - 0.05 * y; });
  PlaneStats a, b, all;
  for (std::size_t i = 0; i < pts.size(); ++i) {
    all.add(pts[i]);
    (i < pts.size() / 2 ? a : b).add(pts[i]);
  }
  a.merge(b);
  CHECK(a.n == all.n);
  CHECK((a.mean() - all.mean()).norm() < 1e-12);
  CHECK((a.covariance() - all.covariance()).norm() < 1e-12);
  // merge with a DIFFERENT ref (b was accumulated about its own first point)
  PlaneStats rev; rev.merge(b);  // rev.ref == b.ref
  PlaneStats first_half;
  for (std::size_t i = 0; i < pts.size() / 2; ++i) first_half.add(pts[i]);
  rev.merge(first_half);
  CHECK((rev.covariance() - all.covariance()).norm() < 1e-12);

  auto batch = batchFit(pts);
  auto f = fitPlane(all, ValidityRule{});
  CHECK(f.valid && f.rank2);
  CHECK((f.center - batch.center).norm() < 1e-9);
  CHECK((f.eigenvalues - batch.eigenvalues).norm() < 1e-9);
  CHECK(std::abs(std::abs(f.normal.dot(batch.normal)) - 1.0) < 1e-9);

  // Numerical cancellation: cluster at a large offset. The shifted sums must
  // match a two-pass covariance; the naive sum_outer/n - mean*mean^T formula
  // must be measurably worse (proves this check can fail).
  const V3 off(1.0e6, -2.0e6, 5.0e5);
  std::vector<V3> far;
  for (const auto& p : pts) far.push_back(p + off);
  PlaneStats fs; V3 s = V3::Zero(); M3 so = M3::Zero();
  for (const auto& p : far) { fs.add(p); s += p; so += p * p.transpose(); }
  Eigen::SelfAdjointEigenSolver<M3> es_shift(fs.covariance());
  V3 m = s / double(far.size());
  Eigen::SelfAdjointEigenSolver<M3> es_naive(so / double(far.size()) - m * m.transpose());
  const V3 truth = batchFit(pts).eigenvalues;  // translation-invariant
  const double err_shift = (es_shift.eigenvalues() - truth).norm();
  const double err_naive = (es_naive.eigenvalues() - truth).norm();
  CHECK(err_shift < 1e-9);
  CHECK(err_naive > 100.0 * err_shift);
}

void checkValidityAndOrientation() {
  // Collinear (rank-1) cell: undefined normal.
  PlaneStats line;
  for (int i = 0; i < 20; ++i) line.add(V3(i * 0.01, 2 * i * 0.01, 1.0));
  auto prod = fitPlane(line, ValidityRule{});
  CHECK(!prod.valid);
  CHECK(!prod.rank2);
  // The R52 ratio-only rule admits it (documents the defect the new rule fixes).
  auto r52 = fitIncrementalPca(line, 0.10);
  CHECK(r52.valid);

  // Orientation: normals point toward the origin for planes on either side.
  auto up = grid(0.0, 0.0, 5, 5, 0.05, [](double, double) { return 1.0; });
  auto dn = grid(0.0, 0.0, 5, 5, 0.05, [](double, double) { return -1.0; });
  PlaneStats su, sd;
  for (const auto& p : up) su.add(p);
  for (const auto& p : dn) sd.add(p);
  auto fu = fitPlane(su, ValidityRule{}), fd = fitPlane(sd, ValidityRule{});
  CHECK(fu.valid && fd.valid);
  CHECK(fu.d >= 0.0 && fd.d >= 0.0);
  CHECK(std::abs(fu.d - 1.0) < 1e-9 && std::abs(fd.d - 1.0) < 1e-9);
  // Sign-invariant comparisons: same physical plane with the opposite normal.
  PlaneFit a = fu, b = fu;
  b.normal = -b.normal; b.d = -b.d; b.center = a.center + V3(0.3, 0.0, 0.0);
  CHECK(planeAngleDeg(a.normal, b.normal) < 1e-6);
  CHECK(planeOffset(a, b) < 1e-9);
  b.center = a.center + V3(0.0, 0.0, 0.1);
  CHECK(std::abs(planeOffset(a, b) - 0.1) < 1e-9);
}

void checkRobustFamilies() {
  Options o; o.leaf = 1.0; o.robust_reservoir = 20; o.robust_ransac_dist = 0.03; o.robust_ransac_iters = 200;
  auto plane = grid(0.05, 0.05, 6, 5, 0.05, [](double, double) { return 1.5; });  // 30 planar points
  std::vector<V3> pts = plane;
  // 8 scattered (non-collinear) gross outliers in the same cell, all > 0.09 m off z=1.5
  for (int i = 0; i < 8; ++i) pts.emplace_back(0.15 + 0.09 * (i % 4), 0.1 + 0.1 * (i / 4), 1.62 + 0.045 * i);

  auto robust = makeBackend("robust_voxel", o);
  robust->insert(0, pts);
  auto sr = robust->snapshot();
  CHECK(!sr.data.empty());
  std::uint64_t accepted = 0;
  for (const auto& p : sr.data) accepted += p.stats.n;
  CHECK(accepted < pts.size());  // outliers rejected

  auto plain = makeBackend("incremental_pca", o);
  plain->insert(0, pts);
  auto sp = plain->snapshot();
  std::uint64_t plain_accepted = 0;
  for (const auto& p : sp.data) plain_accepted += p.stats.n;
  CHECK(plain_accepted == pts.size());

  // A cell holding fewer points than one block must still be represented
  // (R52 dropped every cell with < robust_reservoir points).
  auto few = grid(0.05, 0.05, 4, 4, 0.05, [](double, double) { return 1.5; });  // 16 < 20
  auto robust2 = makeBackend("robust_voxel", o);
  robust2->insert(0, few);
  auto s2 = robust2->snapshot();
  CHECK(s2.patches == 1);
  CHECK(s2.data[0].stats.n == few.size());
}

void checkRankOneBackend() {
  Options o; o.leaf = 1.0;
  std::vector<V3> line;
  for (int i = 0; i < 40; ++i) line.emplace_back(i * 0.01, 0.1, 1.5);
  auto be = makeBackend("incremental_pca", o);
  be->insert(0, line);
  auto s = be->snapshot();
  CHECK(s.patches == 0);
  CHECK(s.cells_total == 1);
  CHECK(s.cells_rank_deficient == 1);
  // R52 rule reproduced by parameters.
  Options r52 = o; r52.plane_eig_max = 1e30; r52.min_secondary_eig = 0.0; r52.max_planarity = 0.10;
  auto be2 = makeBackend("incremental_pca", r52);
  be2->insert(0, line);
  CHECK(be2->snapshot().patches == 1);
}

void checkMergeSplit() {
  Options o; o.leaf = 1.0; o.merge_gap = 1.5;
  auto A = grid(0.05, 0.05, 8, 8, 0.1, [](double, double) { return 1.5; });   // cell x-bin 0
  auto B = grid(1.05, 0.05, 8, 8, 0.1, [](double, double) { return 1.5; });   // cell x-bin 1, coplanar

  for (const char* crit : {"combined_fit", "pairwise"}) {
    Options oc = o; oc.merge_criterion = crit;
    auto m = makeBackend("mergeable_voxel", oc);
    m->insert(0, A); m->insert(0, B);
    auto sm = m->snapshot();
    CHECK(sm.patches == 2);
    CHECK(sm.surfaces == 1);
    CHECK(sm.merges >= 1);
    CHECK(sm.data[0].children.size() == 2);
  }

  // Tilted neighbour must NOT merge.
  auto Bt = grid(1.05, 0.05, 8, 8, 0.1, [](double x, double) { return 1.5 + 0.36 * (x - 1.4); });  // ~20 deg
  auto nm = makeBackend("mergeable_voxel", o);
  nm->insert(0, A); nm->insert(0, Bt);
  auto snm = nm->snapshot();
  CHECK(snm.patches == 2 && snm.surfaces == 2 && snm.merges == 0);

  // Transitive chaining: a gently curved strip. Pairwise single-linkage merges
  // the whole strip into one surface; combined_fit must cut it.
  Options oc = o; oc.leaf = 0.5;
  std::vector<V3> strip = grid(0.0, 0.0, 60, 10, 0.05, [](double x, double) { return 1.0 + 0.05 * x * x; });
  Options opw = oc; opw.merge_criterion = "pairwise";
  Options ocf = oc; ocf.merge_criterion = "combined_fit";
  auto pw = makeBackend("mergeable_voxel", opw);
  auto cf = makeBackend("mergeable_voxel", ocf);
  pw->insert(0, strip); cf->insert(0, strip);
  auto spw = pw->snapshot(), scf = cf->snapshot();
  CHECK(spw.patches == scf.patches && spw.patches >= 5);
  CHECK(spw.surfaces == 1);
  CHECK(scf.surfaces >= 2);

  // Reversibility: merge coplanar A/B, then tilt B's fit with many more
  // points. gaussian_surface must split; mergeable_voxel must not.
  auto tilt = grid(1.0, 0.0, 16, 16, 0.05, [](double x, double) { return 1.5 + 0.466 * (x - 1.375); });  // ~25 deg
  auto g = makeBackend("gaussian_surface", o);
  auto mv = makeBackend("mergeable_voxel", o);
  g->insert(0, A); g->insert(0, B); mv->insert(0, A); mv->insert(0, B);
  CHECK(g->snapshot().surfaces == 1);
  CHECK(mv->snapshot().surfaces == 1);
  g->insert(0, tilt); mv->insert(0, tilt);
  auto sg = g->snapshot(), smv = mv->snapshot();
  CHECK(sg.splits >= 1);
  CHECK(smv.splits == 0);
  CHECK(sg.surfaces > smv.surfaces || sg.patches < smv.patches);

  // The split check depends on the thresholds (it can fail): with lenient
  // thresholds the same data must not split.
  Options lenient = o; lenient.merge_angle_deg = 89.0; lenient.merge_offset = 1e9; lenient.split_planarity_max = 1.0;
  auto gl = makeBackend("gaussian_surface", lenient);
  gl->insert(0, A); gl->insert(0, B); gl->snapshot(); gl->insert(0, tilt);
  CHECK(gl->snapshot().splits == 0);
}

void checkDebiased() {
  auto pts = grid(0.05, 0.05, 8, 8, 0.1, [](double, double) { return 1.5; });

  // poseCovAtBody with R=I, P_RP=0: a(|p|^2 I - p p^T) + b I (independent identity).
  PoseCovContext ctx; ctx.P_RR = 2e-6 * M3::Identity(); ctx.P_PP = 3e-6 * M3::Identity();
  const V3 p(0.3, -0.2, 1.1);
  const M3 expect = 2e-6 * (p.squaredNorm() * M3::Identity() - p * p.transpose()) + 3e-6 * M3::Identity();
  CHECK((poseCovAtBody(ctx, p) - expect).norm() < 1e-15);

  // Zero covariance terms: debiased == plain PCA eigenvalues.
  PlaneStats st;
  for (const auto& q : pts) { st.add(q); st.noteObs(0); }
  auto fp = fitPlane(st, ValidityRule{});
  auto fd0 = fitPlaneDebiased(st, ValidityRule{}, DebiasRule{});
  CHECK(fd0.valid && fp.valid);
  CHECK((fd0.eigenvalues - fp.eigenvalues).norm() < 1e-12);

  // Isotropic sensor variance s2 is subtracted in full from every eigenvalue.
  const double s2 = 1e-4;
  PlaneStats ss;
  for (const auto& q : pts) { ss.add(q, s2 * M3::Identity(), M3::Zero()); ss.noteObs(0); }
  auto fs = fitPlaneDebiased(ss, ValidityRule{}, DebiasRule{});
  CHECK(fs.valid);
  CHECK(std::abs(fs.eigenvalues[2] - (fp.eigenvalues[2] - s2)) < 1e-12);
  CHECK(std::abs(fs.eigenvalues[1] - (fp.eigenvalues[1] - s2)) < 1e-12);
  CHECK(fs.eigenvalues[0] == 0.0);  // clamped, as in production

  // Pose noise is shrunk by (F-1)/F: F=1 -> no correction; F=2 -> half.
  auto poseStats = [&](int frames) {
    PlaneStats t;
    for (std::size_t i = 0; i < pts.size(); ++i) {
      t.add(pts[i], M3::Zero(), poseCovAtBody(ctx, pts[i]));
      t.noteObs(static_cast<std::uint64_t>(i % frames));
    }
    return t;
  };
  auto f1 = fitPlaneDebiased(poseStats(1), ValidityRule{}, DebiasRule{});
  auto f2 = fitPlaneDebiased(poseStats(2), ValidityRule{}, DebiasRule{});
  CHECK((f1.eigenvalues - fp.eigenvalues).norm() < 1e-12);
  CHECK((f2.eigenvalues - fp.eigenvalues).norm() > 1e-7);
  CHECK(f2.eigenvalues.sum() < fp.eigenvalues.sum());

  // Additivity: two halves observed in different frames, merged, equal the
  // one-pass accumulation over both (including the observation-set union).
  PlaneStats a, b, all;
  for (std::size_t i = 0; i < pts.size(); ++i) {
    const M3 pc = poseCovAtBody(ctx, pts[i]);
    const std::uint64_t o = i < pts.size() / 2 ? 0 : 1;
    (o == 0 ? a : b).add(pts[i], s2 * M3::Identity(), pc);
    (o == 0 ? a : b).noteObs(o);
    all.add(pts[i], s2 * M3::Identity(), pc);
    all.noteObs(o);
  }
  a.merge(b);
  CHECK(a.obs.count() == 2 && all.obs.count() == 2);
  auto fm = fitPlaneDebiased(a, ValidityRule{}, DebiasRule{});
  auto fa = fitPlaneDebiased(all, ValidityRule{}, DebiasRule{});
  CHECK((fm.eigenvalues - fa.eigenvalues).norm() < 1e-12);

  // Backend level: the debiased backend runs and differs from PCA on the same input.
  Options o; o.leaf = 1.0; o.debiased = true; o.sensor_var = s2; o.pose = ctx;
  auto deb = makeBackend("incremental_pca", o);
  Options op = o; op.debiased = false;
  auto pca = makeBackend("incremental_pca", op);
  deb->insert(0, pts); pca->insert(0, pts);
  auto sd = deb->snapshot(), sp = pca->snapshot();
  CHECK(sd.patches == 1 && sp.patches == 1);
  CHECK(std::abs(sd.data[0].fit.eigenvalues[2] - (sp.data[0].fit.eigenvalues[2] - s2)) < 1e-9);
}

}  // namespace

int main() {
  checkStats();
  checkValidityAndOrientation();
  checkRobustFamilies();
  checkRankOneBackend();
  checkMergeSplit();
  checkDebiased();
  std::printf("test_stationary_map_stats: all checks passed\n");
  return 0;
}
