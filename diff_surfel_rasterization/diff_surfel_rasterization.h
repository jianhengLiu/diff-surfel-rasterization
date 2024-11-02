#pragma once

#include <torch/torch.h>
#include <tuple>

namespace diff_surfel_rasterization {

class RasterizationSettings : public torch::CustomClassHolder {
public:
  int image_height;
  int image_width;
  float tanfovx;
  float tanfovy;
  torch::Tensor bg;
  float scale_modifier;
  torch::Tensor viewmatrix;
  torch::Tensor projmatrix;
  int sh_degree;
  torch::Tensor campos;
  bool prefiltered;
  bool debug;

  RasterizationSettings() = default;
  RasterizationSettings(int image_height_, int image_width_, float tanfovx_,
                        float tanfovy_, const torch::Tensor &bg_,
                        float scale_modifier_, const torch::Tensor &viewmatrix_,
                        const torch::Tensor &projmatrix_, int sh_degree_,
                        const torch::Tensor &campos_, bool prefiltered_ = false,
                        bool debug_ = false)
      : image_height(image_height_), image_width(image_width_),
        tanfovx(tanfovx_), tanfovy(tanfovy_), bg(bg_),
        scale_modifier(scale_modifier_), viewmatrix(viewmatrix_),
        projmatrix(projmatrix_), sh_degree(sh_degree_), campos(campos_),
        prefiltered(prefiltered_), debug(debug_) {}
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rasterize_gaussians(
    const torch::Tensor &means3D, const torch::Tensor &means2D,
    const torch::Tensor &sh, const torch::Tensor &colors_precomp,
    const torch::Tensor &opacities, const torch::Tensor &scales,
    const torch::Tensor &rotations, const torch::Tensor &cov3Ds_precomp,
    const torch::IValue &raster_settings_ivalue);

class RasterizeGaussiansFunction
    : public torch::autograd::Function<RasterizeGaussiansFunction> {
public:
  static torch::autograd::tensor_list
  forward(torch::autograd::AutogradContext *ctx, const torch::Tensor &means3D,
          const torch::Tensor &means2D, const torch::Tensor &sh,
          const torch::Tensor &colors_precomp, const torch::Tensor &opacities,
          const torch::Tensor &scales, const torch::Tensor &rotations,
          const torch::Tensor &cov3Ds_precomp,
          const torch::IValue &raster_settings_ivalue);

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs);
};

class GaussianRasterizer : public torch::nn::Module {
public:
  torch::IValue raster_settings_ivalue;

  GaussianRasterizer(const RasterizationSettings &raster_settings_);

  torch::Tensor mark_visible(torch::Tensor &positions);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  forward(const torch::Tensor &means3D, const torch::Tensor &means2D,
          const torch::Tensor &opacities,
          const torch::Tensor &shs = torch::Tensor(),
          const torch::Tensor &colors_precomp = torch::Tensor(),
          const torch::Tensor &scales = torch::Tensor(),
          const torch::Tensor &rotations = torch::Tensor(),
          const torch::Tensor &cov3Ds_precomp = torch::Tensor());
};

} // namespace diff_surfel_rasterization
