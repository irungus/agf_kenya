#------------------------------------loading the necessary libries--------------------------------------#
library(tidyverse)
library(devtools)
library(disbayes)
library(tmap)
library(sf)
library(ggspatial)
#------------------------------------simple function----------------------------------------------------#
compute_mean <- function(data){
  stats_summary <-function(x){
    apply(x, 2, mean(x), median(x), sd(x))
  }
  return(stats_summary)
}
#-------------------------------------------------------------------------------------------------------#
#---------------------------------------Kenyan map------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------#

#-------------------------------------Define the data---------------------------------------------------#

county_data <- tibble(
  Region = c("Eastern", "Central", "Rift Valley", "Nyanza", "Western", "Coastal"),
  Counties = list(
    c("Machakos", "Makueni", "Taita Taveta", "Kajiado", "Kitui"),
    c("Kiambu", "Nyandarua", "Nyeri", "Meru", "Tharaka Nithi", "Kirinyaga", "Embu", "Murang’a"),
    c("Elgeyo Marakwet", "Uasin Gishu", "Nandi", "Baringo", "Kericho", "Nakuru", "Narok", "Bomet"),
    c("Nyamira", "Kisii", "Migori", "Homa Bay", "Kisumu", "Siaya"),
    c("Busia", "Bungoma", "Kakamega", "Vihiga", "Trans Nzoia"),
    c("Mombasa", "Kilifi", "Tana River", "Kwale")
  )
)

#---------------------------------Load Kenyan counties shapefile----------------------------------------#

kenya_shp <- st_read("/Users/eomondi/Library/CloudStorage/OneDrive-aphrc.org/Other-files/EAST/gadm41_KEN_shp/gadm41_KEN_1.shp")



#---------------------------------Merge the data with the shapefile-------------------------------------#
#------------------------Create a long format data frame with Region and County-------------------------#

county_long <- county_data %>% unnest(cols = Counties) %>% rename(County = Counties)

#-----------------------------------Join with shapefile data--------------------------------------------#

kenya_shp <- kenya_shp %>% mutate(County = str_to_title(NAME_1)) %>% 
  left_join(county_long, by = "County")

#-------------------------------------Plot the map------------------------------------------------------#

ggplot(data = kenya_shp) + geom_sf(aes(fill = Region), color = "black", size = 0.2) +
  scale_fill_brewer(palette = "Set3", na.value = "grey90", name = "Regions") +
  geom_sf_text(aes(label = NAME_1), size = 2.0, hjust = 1 ) + 
  labs(
    x = "Longitude", 
    y= "Latitude",
    title = "Collection of data by region and counties",
    #subtitle = "counties",
    caption = "Source: GADM & Custom Data"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    legend.position = "right"
  )

# -------------------------Only counties in the county_data---------------------------------------------#
filtered_shp <- kenya_shp %>% filter(!is.na(Region))  


graph1 <- ggplot(data = filtered_shp) +
  geom_sf(aes(fill = Region), color = "black", size = 0.2) +
  scale_fill_brewer(palette = "Set3", name = "Regions") +
  geom_sf_text(aes(label = NAME_1), size = 2.0, hjust = 1 ) + 
  labs(
    x = "Longitude", 
    y= "Latitude",
    #title = "Collection of data by region and counties",
    #subtitle = "counties",
    caption = "Source: GADM & Custom Data"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    legend.position = "right"
  ) + 
  annotation_scale(location = "bl", width_hint = 0.5) +  
  annotation_north_arrow(location = "tr", which_north = "true", 
                         style = north_arrow_fancy_orienteering()) 

print(graph1)


path <- "/Users/eomondi/Library/CloudStorage/OneDrive-aphrc.org/Proposals/Lacuna/code/graph.pdf"

#ggsave(path = path, width = width, height = height, device='tiff', dpi=700)

#tiff("test.tiff", units="in", width=5, height=5, res=300)

ggsave(filename = path, plot = graph1, units="in", width = 5, height = 5, device='pdf', dpi=700)

#----------------------------------------------End------------------------------------------------------#



