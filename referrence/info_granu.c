float info_gran()
{ int i, kk, j, k;
    float ro, dr, dist, cov, ropt, max;
    fprintf(fout1,"     INFORMATION GRANULES  \n");
    for (i=1; i<=c; i++)
    {
        ro=0.0; dr=0.025; max =0.0;
        for (kk=1; kk<=40; kk++)
        {   cov=0.0;
            for(k=1; k<=N_data; k++){
              dist =0.0;
              for (j=1; j<=n_data; j++){
                dist = dist + powf( xx[k][j]-prot[i][j], 2.0)/sigma[j];
              }
              if (dist <=n_data*ro){
                  cov=cov+u[i][k];
              }
            }
            fprintf(fout1, "%f  %f   %f   %f \n", ro, cov, 1-ro, cov*(1-ro));
            if(max<=cov*(1-ro) ) {max = cov*(1-ro); ropt =ro;}
            ro=ro+dr;
        }
        fprintf(fout1, "%d  %f   %f \n", i, ropt, max);
    }

}
