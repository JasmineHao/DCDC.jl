using Pkg
Pkg.activate(".")

using Gadfly, Compose
set_default_plot_size(14cm, 8cm)
plot(sin, 0, 2pi, Guide.annotation(compose(context(),
     Shape.circle([pi/2, 3*pi/2], [1.0, -1.0], [2mm]),
     fill(nothing), stroke("orange"))))


# Another example
using Gadfly, RDatasets
set_default_plot_size(14cm, 8cm)
Dsleep = dataset("ggplot2", "msleep")[[:Vore,:BrainWt,:BodyWt,:SleepTotal]]
DataFrames.dropmissing!(Dsleep)
Dsleep[:SleepTime] = Dsleep[:SleepTotal] .> 8
plot(Dsleep, x=:BodyWt, y=:BrainWt, Geom.point, color=:SleepTime,
  Guide.colorkey(title="Sleep", labels=[">8","≤8"]),
  Scale.x_log10, Scale.y_log10 )
