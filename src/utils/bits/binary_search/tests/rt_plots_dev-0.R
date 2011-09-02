
range<-"5M_1M"
#device<-"GTX460"
#device<-"GTX285"
device<-"9800GT"

t_sm_name<-paste("t_sm_b_search_cuda-",range,".",device,".txt",sep="")
t_gm_name<-paste("t_gm_b_search_cuda-",range,".",device,".txt",sep="")
i_sm_name<-paste("i_sm_b_search_cuda-",range,".",device,".txt",sep="")
i_gm_name<-paste("i_gm_b_search_cuda-",range,".",device,".txt",sep="")
b_name<-paste("b_search_cuda-",range,".",device,".txt",sep="")
s_name<-paste("sort_b_search_cuda-",range,".",device,".txt",sep="")
s_t_gm_name<-paste("sort_t_gm_b_search_cuda-",range,".",device,".txt",sep="")
s_i_gm_name<-paste("sort_i_gm_b_search_cuda-",range,".",device,".txt",sep="")

t_sm <- read.delim(t_sm_name,header=F,sep=",")
t_gm <- read.delim(t_gm_name,header=F,sep=",")
i_sm <- read.delim(i_sm_name,header=F,sep=",")
i_gm <- read.delim(i_gm_name,header=F,sep=",")
b <- read.delim(b_name,header=F,sep=",")
s <- read.delim(s_name,header=F,sep=",")
s_t_gm <- read.delim(s_t_gm_name,header=F,sep=",")
s_i_gm <- read.delim(s_i_gm_name,header=F,sep=",")

#d_s<-unique(t_sm$V1)
#q_s<-unique(t_sm$V2)

d_s = 5000000
q_s = 1000000

out_name<-paste(device,".png",sep="")
png(out_name,width=1000)
for (d in d_s) {
	for (q in q_s) {

		#t_sm_s<-subset(t_sm, V1==d & V2==q & V3<=4095)
		#t_gm_s<-subset(t_gm, V1==d & V2==q & V3<=4095)
		#i_gm_s<-subset(i_gm, V1==d & V2==q & V3<=4095)
		#i_sm_s<-subset(i_sm, V1==d & V2==q & V3<=4095)
		#b_s<-subset(b, V1==d & V2==q & V3<=4095)
		#s_s<-subset(s, V1==d & V2==q & V3<=4095)

		t_sm_s<-subset(t_sm, V1==d & V2==q)
		t_gm_s<-subset(t_gm, V1==d & V2==q)
		i_gm_s<-subset(i_gm, V1==d & V2==q)
		i_sm_s<-subset(i_sm, V1==d & V2==q)
		b_s<-subset(b, V1==d & V2==q)
		s_s<-subset(s, V1==d & V2==q)
		s_t_gm_s<-subset(s_t_gm, V1==d & V2==q)
		s_i_gm_s<-subset(s_i_gm, V1==d & V2==q)

		#bs_s<-subset(bs, V1==d & V2==q)
		#is_s<-subset(is, V1==d & V2==q)
		#ts_s<-subset(ts, V1==d & V2==q)

		x_min<-min(t_sm_s$V3, t_gm_s$V3,i_gm_s$V3,i_sm_s$V3,
				b_s$V3,s_s$V3,s_t_gm_s$V3,s_i_gm_s$V3)
				#bs_s$V3,is_s$V3,ts_s$V3)
		x_max<-max(t_sm_s$V3, t_gm_s$V3, i_gm_s$V3,i_sm_s$V3,
				b_s$V3,s_s$V3,s_t_gm_s$V3,s_i_gm_s$V3)
				#bs_s$V3,is_s$V3,ts_s$V3)
		y_min<-min(t_sm_s$V5, t_gm_s$V5, i_gm_s$V5,i_sm_s$V5,
				b_s$V5,s_s$V5,s_t_gm_s$V5,s_i_gm_s$V5)
				#bs_s$V3,is_s$V5,ts_s$V5)
		y_max<-max(t_sm_s$V5, t_gm_s$V5, i_gm_s$V5,i_sm_s$V5,
				b_s$V5,s_s$V5,s_t_gm_s$V5,s_i_gm_s$V5)
				#bs_s$V3,is_s$V5,ts_s$V5)

		title_string<-paste("Binary Search, Database:",d,
							" Queries:",q,sep="")

		plot(1, 1, type="n",log="x",
			ylim=c(y_min,y_max),xlim=c(x_min,x_max),
			pch=1,
			#col=1,
			bty="n",xlab="Index Size",ylab="Run Time (ms)",
			main=title_string)
			#main="Binary Search, Database:1000000, Queries:1000000" )

		#plot(t_sm_s$V3, t_sm_s$V5, type="o",log="x",
			#ylim=c(y_min,y_max),xlim=c(x_min,x_max),
			#pch=1,
			#col=1,
			#bty="n",xlab="Index Size",ylab="Run Time (ms)",
			#main=title_string)
			#main="Binary Search, Database:1000000, Queries:1000000" )


		lines(b_s$V3, b_s$V5, type="o" ,pch=1)#,col=5)
		lines(s_s$V3, s_s$V5, type="o" ,pch=2)#,col=6)

		lines(i_gm_s$V3, i_gm_s$V5, type="o" ,pch=3)#,col=4)
		lines(i_sm_s$V3, i_sm_s$V5, type="o" ,pch=4)#,col=3)

		lines(t_gm_s$V3, t_gm_s$V5, type="o" ,pch=5)#,col=2)
		lines(t_sm_s$V3, t_sm_s$V5, type="o" ,pch=6)#,col=2)

		lines(s_i_gm_s$V3, s_i_gm_s$V5, type="o" ,pch=7)#,col=8)
		lines(s_t_gm_s$V3, s_t_gm_s$V5, type="o" ,pch=8)#,col=7)

		b_l<-paste("No Index (",device,")",sep="")
		s_l<-paste("Sorted, No Index (",device,")",sep="")
		t_sm_l<-paste("Tree, Shared Mem(",device,")",sep="")
		t_gm_l<-paste("Tree, Global Mem(",device,")", sep="")
		l_sm_l<-paste("List, Shared Mem(",device,")",sep="")
		l_gm_l<-paste("List, Global Mem(",device,")",sep="")
		s_t_gm_l<-paste("Sorted, Tree, Global Mem(",device,")",sep="")
		s_i_gm_l<-paste("Sorted, List, Global Mem(",device,")",sep="")

		legend("left", c( 
						  b_l,
						  s_l,
						  l_gm_l,
						  l_sm_l,
						  t_gm_l,
						  t_sm_l,
						  s_i_gm_l,
						  s_t_gm_l
						),
				pch=1:8,
				#col=1:8,
				bty="n",
				lty=1)
	}
}
dev.off()


