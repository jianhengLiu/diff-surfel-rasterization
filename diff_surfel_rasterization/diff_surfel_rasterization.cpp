#include "diff_surfel_rasterization.h"

#include "rasterize_points.h"

namespace diff_surfel_rasterization {

inline GaussianRasterizer::GaussianRasterizer(
    const RasterizationSettings &raster_settings_) {
  raster_settings_ivalue = torch::IValue(
      torch::make_intrusive<RasterizationSettings>(raster_settings_));
}

inline torch::Tensor
GaussianRasterizer::markVisible(const torch::Tensor &positions) {
  torch::NoGradGuard no_grad;
  // Placeholder for _C.mark_visible function
  // Implement the visibility check here
  torch::Tensor visible;
  return visible;
}

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
GaussianRasterizer::forward(
    const torch::Tensor &means3D, const torch::Tensor &means2D,
    const torch::Tensor &opacities, const torch::Tensor &shs,
    const torch::Tensor &colors_precomp, const torch::Tensor &scales,
    const torch::Tensor &rotations, const torch::Tensor &cov3Ds_precomp) {
  if ((shs.defined() && colors_precomp.defined()) ||
      (!shs.defined() && !colors_precomp.defined())) {
    throw std::runtime_error(
        "Please provide exactly one of either SHs or precomputed colors!");
  }
  if (((!scales.defined() || !rotations.defined()) &&
       !cov3Ds_precomp.defined()) ||
      ((scales.defined() || rotations.defined()) && cov3Ds_precomp.defined())) {
    throw std::runtime_error(
        "Please provide exactly one of either scale/rotation pair or "
        "precomputed 3D covariance!");
  }
  torch::Tensor shs_ =
      shs.defined() ? shs : torch::empty({0}, torch::device(torch::kCUDA));
  torch::Tensor colors_precomp_ =
      colors_precomp.defined() ? colors_precomp
                               : torch::empty({0}, torch::device(torch::kCUDA));
  torch::Tensor scales_ = scales.defined()
                              ? scales
                              : torch::empty({0}, torch::device(torch::kCUDA));
  torch::Tensor rotations_ =
      rotations.defined() ? rotations
                          : torch::empty({0}, torch::device(torch::kCUDA));
  torch::Tensor cov3Ds_precomp_ =
      cov3Ds_precomp.defined() ? cov3Ds_precomp
                               : torch::empty({0}, torch::device(torch::kCUDA));

  return rasterize_gaussians(means3D, means2D, shs_, colors_precomp_, opacities,
                             scales_, rotations_, cov3Ds_precomp_,
                             raster_settings_ivalue);
}

// Implement RasterizeGaussiansFunction forward and backward methods here
// Note: You'll need to interface with your C++/CUDA rasterization backend (_C
// module in Python)

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_gaussians(const torch::Tensor &means3D, const torch::Tensor &means2D,
                    const torch::Tensor &sh,
                    const torch::Tensor &colors_precomp,
                    const torch::Tensor &opacities, const torch::Tensor &scales,
                    const torch::Tensor &rotations,
                    const torch::Tensor &cov3Ds_precomp,
                    const torch::IValue &raster_settings_ivalue) {
  auto results = RasterizeGaussiansFunction::apply(
      means3D, means2D, sh, colors_precomp, opacities, scales, rotations,
      cov3Ds_precomp, raster_settings_ivalue);
  return std::make_tuple(results[0], results[1], results[2]);
}

inline torch::autograd::tensor_list RasterizeGaussiansFunction::forward(
    torch::autograd::AutogradContext *ctx, const torch::Tensor &means3D,
    const torch::Tensor &means2D, const torch::Tensor &sh,
    const torch::Tensor &colors_precomp, const torch::Tensor &opacities,
    const torch::Tensor &scales, const torch::Tensor &rotations,
    const torch::Tensor &cov3Ds_precomp,
    const torch::IValue &raster_settings_ivalue) {

  // Call the C++/CUDA rasterizer (Placeholder for _C.rasterize_gaussians)
  int num_rendered;
  torch::Tensor color, depth, radii, geomBuffer, binningBuffer, imgBuffer;
  auto raster_settings =
      raster_settings_ivalue.toCustomClass<RasterizationSettings>();
  std::tie(num_rendered, color, depth, radii, geomBuffer, binningBuffer,
           imgBuffer) =
      RasterizeGaussiansCUDA(
          raster_settings->bg, means3D, colors_precomp, opacities, scales,
          rotations, raster_settings->scale_modifier, cov3Ds_precomp,
          raster_settings->viewmatrix, raster_settings->projmatrix,
          raster_settings->tanfovx, raster_settings->tanfovy,
          raster_settings->image_height, raster_settings->image_width, sh,
          raster_settings->sh_degree, raster_settings->campos,
          raster_settings->prefiltered, raster_settings->debug);

  // Save context for backward
  ctx->saved_data["raster_settings_ivalue"] = raster_settings_ivalue;
  ctx->saved_data["num_rendered"] = num_rendered;
  ctx->save_for_backward({colors_precomp, means3D, scales, rotations,
                          cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer,
                          imgBuffer});

  return {color, radii, depth};
}

inline torch::autograd::tensor_list RasterizeGaussiansFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::tensor_list grad_outputs) {
  auto raster_settings_ivalue =
      ctx->saved_data["raster_settings_ivalue"].toIValue();
  auto raster_settings =
      raster_settings_ivalue.toCustomClass<RasterizationSettings>();
  auto num_rendered = ctx->saved_data["num_rendered"].toInt();
  auto saved = ctx->get_saved_variables();

  auto colors_precomp = saved[0];
  auto means3D = saved[1];
  auto scales = saved[2];
  auto rotations = saved[3];
  auto cov3Ds_precomp = saved[4];
  auto radii = saved[5];
  auto sh = saved[6];
  auto geomBuffer = saved[7];
  auto binningBuffer = saved[8];
  auto imgBuffer = saved[9];

  auto grad_out_color = grad_outputs[0];
  auto grad_radii = grad_outputs[1];
  auto grad_depth = grad_outputs[2];

  // Call the C++/CUDA backward function (Placeholder for
  // _C.rasterize_gaussians_backward)
  torch::Tensor grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D;
  torch::Tensor grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations;

  std::tie(grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D,
           grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations) =
      RasterizeGaussiansBackwardCUDA(
          raster_settings->bg, means3D, radii, colors_precomp, scales,
          rotations, raster_settings->scale_modifier, cov3Ds_precomp,
          raster_settings->viewmatrix, raster_settings->projmatrix,
          raster_settings->tanfovx, raster_settings->tanfovy, grad_out_color,
          grad_depth, sh, raster_settings->sh_degree, raster_settings->campos,
          geomBuffer, num_rendered, binningBuffer, imgBuffer,
          raster_settings->debug);

  // Return gradients
  torch::autograd::tensor_list grads = {
      grad_means3D,        grad_means2D,        grad_sh,
      grad_colors_precomp, grad_opacities,      grad_scales,
      grad_rotations,      grad_cov3Ds_precomp, torch::Tensor()};

  return grads;
}
} // namespace diff_surfel_rasterization