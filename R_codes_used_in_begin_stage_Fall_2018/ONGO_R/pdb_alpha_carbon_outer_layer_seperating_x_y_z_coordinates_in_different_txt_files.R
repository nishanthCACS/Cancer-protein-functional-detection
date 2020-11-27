# to clear the work space
rm(list=ls())
library(bio3d)
library(plot3D)
library(plotly)
library(stringr)
# install.packages("plot3D", dependencies=TRUE)

pdb_file_to_property <- function(name_pdb)
{
  name_pdb ="121p.pdb"
  pdb <- read.pdb(name_pdb)
  # print(pdb)
  ca.inds<-atom.select(pdb,"calpha")
  # plot.bio3d(pdb)
  #head(pdb$xyz)
  head(pdb$atom[ca.inds$atom,])
  dim(pdb$atom[ca.inds$atom,])[1]
  # take consider only the xyz coordinates of alpha carbon
  xyz <- pdb$atom[ca.inds$atom,][9:11]
  # ctr+L to clear the screen
  
  # the xyz function change the cartesian coordinate to polar coordinate
  xyz_polar <- function(xyz)
  {
    # the xyz function change the cartesian coordinate to polar coordinate
    x<-xyz[1:dim(pdb$atom[ca.inds$atom,])[1],1]
    y<-xyz[1:dim(pdb$atom[ca.inds$atom,])[1],2]
    z<-xyz[1:dim(pdb$atom[ca.inds$atom,])[1],3]
    #  find the center of gravity from the x-y-z coordinates
    g<-c(mean(x),mean(y),mean(z))
    # change the xyz coordinates in matrix form
    xyz_num=cbind(x,y,z)
    
    
    # this will cetralize the center of gravity in coordinate 0,0,0
    d_cal_init <- matrix(NA, nrow = dim(xyz_num)[1], ncol = 3)
    for(i in 1:dim(xyz_num)[1]) {
      # find the difference between the center of gravity and the atom in x-y-z coordinates
      d_cal_init[i,1:3]=xyz_num[i,1:3]-g
    }
    
    #first findout the minimum position to move
    m<-c(min(x),min(y),min(z))
    # this will linearly change the position
    d_final <- matrix(NA, nrow = dim(xyz_num)[1], ncol = 3)
    for(i in 1:dim(xyz_num)[1]) {
      # find the difference between the center of gravity and the atom in x-y-z coordinates
      d_final[i,1:3]=xyz_num[i,1:3]-m
    }
    
    
    cart2pol <- function(x, y)
    {
      r <- sqrt(x^2 + y^2)
      if (x>0){
        t <- atan(y/x)# check it is in the all region quadrant
      }
      else if(x<0 & y>0){
        t <- c(pi+atan(y/x))# check it is in the second
      }
      else{
        t <- c(-pi+atan(y/x))# check it is in the second
      }
      c(r,t)
    }
    # first create an empty matrices to hold the data
    xy_polar<-matrix(NA, nrow = dim(xyz_num)[1], ncol = 2)#here first coloumn contain the radius and the second coloumn contain angle
    for(i in 1:dim(xyz_num)[1]) {
      # find the radius of x-y coordinate
      xy_polar[i,1]=cart2pol(d_cal_init[i,1],d_cal_init[i,2])[1]
      xy_polar[i,2]=cart2pol(d_cal_init[i,1],d_cal_init[i,2])[2]
    }
    
    #xy to z coordinate angle
    xy_z_polar<-matrix(NA, nrow = dim(xyz_num)[1], ncol = 2)#here first coloumn contain the radius and the second coloumn contain angle
    for(i in 1:dim(xyz_num)[1]) {
      # find the radius of xy-to z coordinate
      #take the xy radius
      xy_z_polar[i,1]=cart2pol(xy_polar[i,1],d_cal_init[i,3])[1]
      xy_z_polar[i,2]=cart2pol(xy_polar[i,1],d_cal_init[i,3])[2]
    }
    
    return(list(final_polar=cbind(xy_z_polar[1:dim(xyz)[1],1],xy_polar[1:dim(xyz)[1],2],xy_z_polar[1:dim(xyz)[1],2]),d_final=d_final))
    
  }
  polar_function_object<-xyz_polar(xyz)#to find out the normalized centered coordinates
  final_polar= polar_function_object$final_polar
  d_final=polar_function_object$d_final
  
  atom_angle_range <- function(angle_chk,resolution)
  {
    #this function will give the range where the atom belongs
    range_xy_angle <- matrix(NA, nrow = length(angle_chk), ncol = 1)
    angle_list_rad=seq(from=-180, to=180, by=resolution)*pi/180
    for(i in 1:dim(final_polar)[1]){
      for(j in 1:length(angle_list_rad)-1){
        if(angle_list_rad[j]<=angle_chk[i] && angle_chk[i]<angle_list_rad[j+1]){
          range_xy_angle[i]=j
        }
      }
    }
    range_xy_angle
  }
  
  surface_select_atom <- function(final_polar,resolution){
    # this function find out the surface atoms
    angle_xy=final_polar[1:dim(final_polar)[1],2]
    angle_xy_z=final_polar[1:dim(final_polar)[1],3]
    
    range_of_xy=atom_angle_range(angle_xy,resolution)
    range_of_xy_z=atom_angle_range(angle_xy_z,resolution)
    radius=final_polar[1:dim(final_polar)[1],1]
    # then just check the
    checked_range_element <- list();
    # selected_atom <- list();
    selected_atom<-c()
    for(i in 1:length(range_of_xy)){
      r=radius[i]
      s=i#to hold the selected item in surface
      if(is.element(i, checked_range_element)){
      }
      else{
        checked_range_element <- rbind(selected_atom, i)
        for (j in 1:length(range_of_xy)) {
          if (i != j) {
            if (range_of_xy[i] == range_of_xy[j]) {
              if (range_of_xy_z[i] == range_of_xy_z[j]) {
                checked_range_element <- rbind(selected_atom, j)
                if (r < radius[j]) {
                  r = radius[j]
                  s = j
                }
              }
            }
          }
        }
        # selected_atom<- c(selected_atom,s);
        selected_atom<-rbind(selected_atom,s)
      }
    }
    unique(selected_atom)
  }
  
  resolution=36# angle degree of resolution
  angle_range=seq(from=-180, to=180, by=resolution)
  selected_calphas=surface_select_atom(final_polar,resolution)# thsi contain the surface selected atoms
  m=selected_calphas
  # length(m)
  
  # the xyz function change the cartesian coordinate to polar coordinate
  x<-xyz[1:dim(pdb$atom[ca.inds$atom,])[1],1]
  y<-xyz[1:dim(pdb$atom[ca.inds$atom,])[1],2]
  z<-xyz[1:dim(pdb$atom[ca.inds$atom,])[1],3]
  # change the xyz coordinates in matrix form
  xyz_num=cbind(x,y,z)
  s_x=xyz_num[m,1]
  s_y=xyz_num[m,2]
  s_z=xyz_num[m,3]
  
  # scatter3D(x, y, z,main='earlier')
  # sprintf("normal mean alpha carbon ")
  # mean(x)
  # mean(y)
  # mean(z)
  # scatter3D(s_x, s_y, s_z,main='only surface')
  # sprintf("surface mean alpha carbon ")
  # mean(s_x)
  # mean(s_y)
  # mean(s_z)
  # resdue--------------
  ca.inds<-atom.select(pdb,"calpha")
  dim(pdb$atom[ca.inds$atom,])[1]
  # To check the residues
  aa3<-pdb$atom$resid[ca.inds$atom]
  aminoacids_calpha<-aa321(aa3)# change the format to one letter
  # to find out the surface atoms residues -----------------------------
  sur_res<-matrix(NA, nrow = length(selected_calphas), ncol = 1)
  sur_res_cor<-matrix(NA, nrow = length(selected_calphas), ncol = 3)
  for(i in 1:length(selected_calphas)){
    sur_res[i]=aminoacids_calpha[selected_calphas[i]]
    # sur_res_cor[i,1:3]=xyz_num[selected_calphas[i],1:3]
    sur_res_cor[i,1:3]= round(2*d_final[selected_calphas[i],1:3],0)
    #take shifted coordinate to [0,0,0] using i-min(i) , j-min(j), and k-min(k)
    # multiply each coordinate by 2 (just for increasing the resolution) and then round them to decimal numbers.
  }
  # creating results for x-y coordinates
  # first create empty matrx for store the results
  
  aminoacids_property_assign <- function(sur_res,sur_res_cor,condition)
  {
    # function to take the amino acids and assign the property here you give the condition as "xy" or "yz" or "xz"
    # first create an empty matrices to hold the properties of alpha carbon
    if (condition=="xy"){
      property<-matrix(NA, nrow = length(sur_res), ncol = 18, byrow = TRUE,dimnames = list(c(),
                                                                                           c("charged(side chain can make salt bridges)",
                                                                                             "Polar(usually participate in hydrogen bonds as proton donnars & acceptors)",	
                                                                                             "Hydrophobic(normally burried inside the protein core)","Hydrophobic","Moderate",
                                                                                             "Hydrophillic","polar","Aromatic","Aliphatic","Acid","Basic","negative charge","Neutral",
                                                                                             "positive charge","Pka_NH2","P_ka_COOH","x","y")))}
    
    else if (condition=="yz"){
      property<-matrix(NA, nrow = length(sur_res), ncol = 18, byrow = TRUE,dimnames = list(c(),
                                                                                           c("charged(side chain can make salt bridges)",
                                                                                             "Polar(usually participate in hydrogen bonds as proton donnars & acceptors)",	
                                                                                             "Hydrophobic(normally burried inside the protein core)","Hydrophobic","Moderate",
                                                                                             "Hydrophillic","polar","Aromatic","Aliphatic","Acid","Basic","negative charge","Neutral",
                                                                                             "positive charge","Pka_NH2","P_ka_COOH","y","z")))}
    
    
    else if (condition=="xz"){
      property<-matrix(NA, nrow = length(sur_res), ncol = 18, byrow = TRUE,dimnames = list(c(),
                                                                                           c("charged(side chain can make salt bridges)",
                                                                                             "Polar(usually participate in hydrogen bonds as proton donnars & acceptors)",	
                                                                                             "Hydrophobic(normally burried inside the protein core)","Hydrophobic","Moderate",
                                                                                             "Hydrophillic","polar","Aromatic","Aliphatic","Acid","Basic","negative charge","Neutral",
                                                                                             "positive charge","Pka_NH2","P_ka_COOH","x","z")))
      
      
    }
    for(i in 1:length(sur_res)) {
      # find the radius of xy-to z coordinate
      #take the xy radius
      if (condition=="xy"){
        property[i,17:18]=sur_res_cor[i,1:2]}
      else if (condition=="yz"){
        property[i,17:18]=sur_res_cor[i,2:3]}
      else if (condition=="xz"){
        property[i,17]=sur_res_cor[i,1]
        property[i,18]=sur_res_cor[i,3]
      }
      
      if (aminoacids_calpha[i]=="M"){
        # print(aminoacids_calpha[i])
        property[i,1:16]=c(0,1,0,0,1,0,1,0,0,0,0,0,1,0,0.207070707,0.079276773)
        
      }
      else if(aminoacids_calpha[i]=="R"){
        property[i,1:16]=c(1,0,0,0,0,1,0,0,0,0,1,0,0,1,0.146464646,0.065368567)
      }
      else if(aminoacids_calpha[i]=="K"){
        property[i,1:16]=c(1,0,0,0,0,1,0,0,0,0,1,0,0,1,0.747474747,1)
      }
      else if(aminoacids_calpha[i]=="D"){
        property[i,1:16]=c(1,0,0,0,0,1,0,0,0,1,0,1,0,0,0.404040404,0.02364395)
      }
      else if(aminoacids_calpha[i]=="E"){
        property[i,1:16]=c(1,0,0,0,0,1,0,0,0,1,0,1,0,0,0.439393939,0.066759388)
      }
      else if(aminoacids_calpha[i]=="Q"){
        property[i,1:16]=c(0,1,0,0,0,1,1,0,0,0,0,0,1,0,0.166666667,0.063977747)
      }
      else if(aminoacids_calpha[i]=="N"){
        property[i,1:16]=c(0,1,0,0,0,1,1,0,0,0,0,0,1,0,0,0.043115438)
      }
      else if(aminoacids_calpha[i]=="H"){
        property[i,1:16]=c(0,1,0,0,1,0,0,0,0,0,1,0,0,1,0.085858586,0.009735744)
      }
      else if(aminoacids_calpha[i]=="S"){
        property[i,1:16]=c(0,1,0,0,0,1,1,0,0,0,0,0,1,0,0.176767677,0.069541029)
      }
      else if(aminoacids_calpha[i]=="T"){
        property[i,1:16]=c(0,1,0,0,0,1,1,0,0,0,0,0,1,0,0.161616162,0.061196106)
      }
      else if(aminoacids_calpha[i]=="Y"){
        property[i,1:16]=c(0,1,0,1,0,0,0,1,0,0,0,0,1,0,0.156565657,0.068150209)
      }
      else if(aminoacids_calpha[i]=="C"){
        property[i,1:16]=c(0,1,0,0,1,0,1,0,0,0,0,0,1,0,1,0)
      }
      else if(aminoacids_calpha[i]=="W"){
        property[i,1:16]=c(0,1,0,1,0,0,0,1,0,0,0,0,1,0,0.297979798,0.093184979)
      }
      else if(aminoacids_calpha[i]=="A"){
        property[i,1:16]=c(0,0,1,1,0,0,0,0,1,0,0,0,1,0,0.54040404,0.089012517)
      }
      else if(aminoacids_calpha[i]=="I"){
        property[i,1:16]=c(0,0,1,1,0,0,0,0,1,0,0,0,1,0,0.484848485,0.084840056)
      }
      else if(aminoacids_calpha[i]=="L"){
        property[i,1:16]=c(0,0,1,1,0,0,0,0,1,0,0,0,1,0,0.404040404,0.090403338)
      }
      else if(aminoacids_calpha[i]=="F"){
        property[i,1:16]=c(0,0,1,1,0,0,0,1,0,0,0,0,1,0,0.222222222,0.121001391)
      }
      else if(aminoacids_calpha[i]=="V"){
        property[i,1:16]=c(0,0,1,1,0,0,0,0,1,0,0,0,1,0,0.464646465,0.080667594)
      }
      else if(aminoacids_calpha[i]=="P"){
        property[i,1:16]=c(0,0,1,1,0,0,0,0,0,0,0,0,1,0,0.909090909,0.038942976)
      }
      else if(aminoacids_calpha[i]=="G"){
        property[i,1:16]=c(0,0,1,1,0,0,0,0,0,0,0,0,1,0,0.404040404,0.087621697)
      }
      
    }
    property
  }
  # property=aminoacids_property_assign(sur_res,sur_res_cor)
  return(list(property_xy=aminoacids_property_assign(sur_res,sur_res_cor,"xy"),property_yz=aminoacids_property_assign(sur_res,sur_res_cor,"yz"),property_xz=aminoacids_property_assign(sur_res,sur_res_cor,"xz")))
  
  # sur_res_cor
  # xyz
}
# p1=pdb_file_to_property("1A07.pdb")
# write.table(p1, "c:/mydata.txt", sep="\t")
#change working directory----------------------------------------
# getwd()
# setwd("C:/Users/User/Documents/R_works/genome_2_phenome")
getwd()
# list.files()
# use the list of files to crate the o/p
# setwd(path.expand("D:\projects\UL projects\PDB-file-R-language") )
# setwd("")
# getwd()
# pdb_ids_obj <- read.table("canser-3.csv",sep=",", header=FALSE)
pdb_ids_obj <- read.table("ONGO_selected_PDBs.csv",sep=",", header=FALSE)
pdb_ids_obj_2<-pdb_ids_obj[1:dim(pdb_ids_obj)[1],1]
# pdb_ids=toString(pdb_ids_obj_2[1])
pdb_ids<-matrix(NA, nrow = length(pdb_ids_obj_2), ncol = 1, byrow = TRUE)
pdb_ids_naming<-matrix(NA, nrow = length(pdb_ids_obj_2), ncol = 1, byrow = TRUE)
#
for (i in 1:length(pdb_ids_obj_2)){
  pdb_ids_naming[i]<-toString(pdb_ids_obj_2[i])
}

pdb_ids_obj <- read.table("ONGO_selected_PDBs_for_R.csv",sep=",", header=FALSE)
pdb_ids_obj_2<-pdb_ids_obj[1:dim(pdb_ids_obj)[1],1]
for (i in 1:length(pdb_ids_obj_2)){
  pdb_ids[i]<-toString(pdb_ids_obj_2[i])
}


# pdb_ids_chk<-pdb_ids[2501:3000]
i=1
# write.table(pdb_ids_chk, file="pdb_ids_chk.txt",sep = ",",row.names = F,col.names = F)
for (ids in pdb_ids[1:1258]){
  p1<-pdb_file_to_property(ids)
  # p1<-pdb_file_to_property(ids)
  property_object<-pdb_file_to_property(ids)#to find out the normalized centered coordinates
  property_xy= property_object$property_xy
  property_yz= property_object$property_yz
  property_xz= property_object$property_xz
  print(paste(pdb_ids_naming[i],".pdb"))
  print(paste("Id number in progress: ",i))
  # p2<-p1[1,1:16]
  
  write.table(property_xy, file=str_c(pdb_ids_naming[i],'_xy',".txt"),sep = ",",row.names = F)
  write.table(property_yz, file=str_c(pdb_ids_naming[i],'_yz',".txt"),sep = ",",row.names = F)
  write.table(property_xz, file=str_c(pdb_ids_naming[i],'_xz',".txt"),sep = ",",row.names = F)
  # fileConn<-file(str_c(ids,".txt"))
  # writeLines(pdb_file_to_property(ids), fileConn)
  # close(fileConn)
  i=i+1
}