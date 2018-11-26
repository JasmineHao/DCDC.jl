using Weave

path = "C:\\Users\\Jasmine\\.julia\\dev\\DCDC\\doc"
weave(joinpath(path,"BandwidthSelection.jmd"),doctype="pandoc",out_path=path)
