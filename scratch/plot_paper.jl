
using Gadfly
using Plotly
using Plots
gr()
# Plots
# http://docs.juliaplots.org/latest/examples/gr/#histogram
#_______________________________________________________________
plot()                                    # empty Plot object
plot(4)                                   # initialize with 4 empty series
plot(rand(10))                            # 1 series... x = 1:10
plot(rand(10,5))                          # 5 series... x = 1:10
plot(rand(10), rand(10),lab="scatter")                  # 1 series
plot(rand(10,5), rand(10))                # 5 series... y is the same for all
plot(sin, rand(10))                       # y = sin(x)
plot(rand(10), sin)                       # same... y = sin(x)
plot([sin,cos], 0:0.1:π)                  # 2 series, sin(x) and cos(x)
plot([sin,cos], 0, π)                     # sin and cos on the range [0, π]
plot(1:10, Any[rand(10), sin])            # 2 series: rand(10) and map(sin,x)

# scatter
my_plot = plot([scatter(x=[1,2], y=[3,4])], Layout(title="My plot"))

#plot(dataset("Ecdat", "Airline"), :Cost)  # the :Cost column from a DataFrame... must import StatPlots

gr()
#Fake data
plot(Plots.fakedata(50, 5), w=3)
#Another fake data
y = rand(100)
plot(0:10:100, rand(11, 4), lab="lines", w=3, palette=:grays, fill=0, α=0.6)
scatter!(y, zcolor=abs.(y .- 0.5), m=(:heat, 0.8, Plots.stroke(1, :green)), ms=10 * abs.(y .- 0.5) .+ 4, lab="grad")

#statistics
using Statistics
y = rand(20, 3)
plot(y, xaxis=("XLABEL", (-5, 30), 0:2:20, :flip), background_color=RGB(0.2, 0.2, 0.2), leg=false)
hline!(mean(y, dims=1) + rand(1, 3), line=(4, :dash, 0.6, [:lightgreen :green :darkgreen]))
vline!([5, 10])
title!("TITLE")
yaxis!("YLABEL", :log10)

# Hist
histogram(randn(1000), bins=:scott, weights=repeat(1:5, outer=200))

#Histogram2D
histogram2d(randn(10000), randn(10000), nbins=20)
# Line styles
styles = filter((s->begin
                s in Plots.supported_styles()
            end), [:solid, :dash, :dot, :dashdot, :dashdotdot])
styles = reshape(styles, 1, length(styles))
n = length(styles)
y = cumsum(randn(20, n), dims=1)
plot(y, line=(5, styles), label=map(string, styles), legendtitle="linestyle")

# Marker type
markers = filter((m->begin
                m in Plots.supported_markers()
            end), Plots._shape_keys)
markers = reshape(markers, 1, length(markers))
n = length(markers)
x = (range(0, stop=10, length=n + 2))[2:end - 1]
y = repeat(reshape(reverse(x), 1, :), n, 1)
scatter(x, y, m=(8, :auto), lab=map(string, markers), bg=:linen, xlim=(0, 10), ylim=(0, 10))

#Plotly Scatter
#_______________________________________________________________

using Plotly
trace1 = [
  "x" => [52698, 43117],
  "y" => [53, 31],
  "mode" => "markers",
  "name" => "North America",
  "text" => ["United States", "Canada"],
  "marker" => [
    "color" => "rgb(164, 194, 244)",
    "size" => 12,
    "line" => [
      "color" => "white",
      "width" => 0.5
    ]
  ],
  "type" => "scatter"
]
trace2 = [
  "x" => [39317, 37236, 35650, 30066, 29570, 27159, 23557, 21046, 18007],
  "y" => [33, 20, 13, 19, 27, 19, 49, 44, 38],
  "mode" => "markers",
  "name" => "Europe",
  "text" => ["Germany", "Britain", "France", "Spain", "Italy", "Czech Rep.", "Greece", "Poland"],
  "marker" => [
    "color" => "rgb(255, 217, 102)",
    "size" => 12,
    "line" => [
      "color" => "white",
      "width" => 0.5
    ]
  ],
  "type" => "scatter"
]
trace3 = [
  "x" => [42952, 37037, 33106, 17478, 9813, 5253, 4692, 3899],
  "y" => [23, 42, 54, 89, 14, 99, 93, 70],
  "mode" => "markers",
  "name" => "Asia/Pacific",
  "text" => ["Australia", "Japan", "South Korea", "Malaysia", "China", "Indonesia", "Philippines", "India"],
  "marker" => [
    "color" => "rgb(234, 153, 153)",
    "size" => 12,
    "line" => [
      "color" => "white",
      "width" => 0.5
    ]
  ],
  "type" => "scatter"
]
trace4 = [
  "x" => [19097, 18601, 15595, 13546, 12026, 7434, 5419],
  "y" => [43, 47, 56, 80, 86, 93, 80],
  "mode" => "markers",
  "name" => "Latin America",
  "text" => ["Chile", "Argentina", "Mexico", "Venezuela", "Venezuela", "El Salvador", "Bolivia"],
  "marker" => [
    "color" => "rgb(142, 124, 195)",
    "size" => 12,
    "line" => [
      "color" => "white",
      "width" => 0.5
    ]
  ],
  "type" => "scatter"
]
data = [trace1, trace2, trace3, trace4]
layout = [
  "title" => "Quarter 1 Growth",
  "xaxis" => [
    "title" => "GDP per Capita",
    "showgrid" => false,
    "zeroline" => false
  ],
  "yaxis" => [
    "title" => "Percent",
    "showline" => false
  ]
]
response = Plotly.plot(data, ["layout" => layout, "filename" => "line-style", "fileopt" => "overwrite"])
plot_url = response["url"]
