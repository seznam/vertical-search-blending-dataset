#!/usr/bin/awk -f

#FILE         eval_SNIPS.awk
#AUTHOR       Pavel Prochazka pavel.prochazka@firma.seznam.cz

#Copyright (c) 2019 Seznam.cz, a.s.
#All rights reserved.

BEGIN{
    FS = "\t"
}
{
    M=NF-9; # Number of methods to eval
    pos=$9;
    if(pos==0){ # Null propensities and metrics
        prop0 = 1
        for(i=0;i<M+2;i++){prop1[i]=1}
        click = -1
        hclick = -1
        vertclick = -1
        #printf("\n")
    }
    pi0=$8;prop0 = prop0 * pi0;if(prop0 < 1e-5){prop0=1e-5} # prop

    for(i=0;i<M;i++){new[i]=$(10+i)-1}
    o_act=$7
    num_act=$6
    for(i=0;i<M;i++) { # loop over methods
        new_act=new[i]
        if(new_act!=o_act){
            prop1[i] = 0
        }
    }
    #print pos, $7, $10-1, $11-1, $12-1, prop1[0], prop1[1], prop1[2], prop0
    # add Random policy
    i = M
    if(num_act<=1){pi1=1}
    if(num_act>1){
        pi1 = 1/num_act
    }
    prop1[i] = prop1[i] * pi1

    # add Logging policy
    i = M + 1
    if(num_act<=1){pi1=1}
    if(num_act>1){
        pi1 = pi0
    }
    prop1[i] = prop1[i] * pi1

    if($1 > 0){click=1}
    if($2 > 0){hclick=pos}
    if($3 > 0){vertclick=1}
    for(i=0;i<M+2;i++){
        c = prop1[i] / prop0
        norm_POS[pos, i] += c
        IPS_click[pos, i] += c * (click>=0)
        IPS_hclick[pos, i] += c * (hclick>=0)
        if(hclick>=0){IPS_ndcg[pos, i] += c * (log(2) / log(hclick+2))}
        IPS_vertclick[pos, i] += c * (vertclick>=0)
    }
    cnt_POS[pos] += 1
}END{
    printf("pos\tC\tcnt\tclick\thappy_click\tndcg\tclick_on_vert\n")
    for(i=0;i<M+2;i++){
        printf("method %d\n", i)
        for(pos=0;pos<14;pos++){
            if(cnt_POS[pos]==0){break}
            np = norm_POS[pos, i]
            nc = np/cnt_POS[pos]
            printf("%d\t%.5f\t%d", pos, nc, cnt_POS[pos])
            if(nc>0){
               printf("\t%.5f\t%.5f\t%.5f\t%.5f", IPS_click[pos, i]/np, IPS_hclick[pos, i]/np, IPS_ndcg[pos, i]/np,IPS_vertclick[pos, i]/np)
            }
            printf("\n")
        }
        printf("\n")
        printf("\n")
    }
}
