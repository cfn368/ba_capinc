foreach world in "" _OPEN {
	forval yyyy = 1963/2023{
		* 1. CREATE MATRIX D FROM THE MAKE DATASET (see TotalRequirementsDerivation.pdf
		use "$MyPath/save/MAKE_`yyyy'", clear
		drop VOther VUsed
		mkmat V*, matrix(V) rownames(code)
		mata: V = st_matrix("V")  
		mata: D = V :/ colsum(V)
		mata: st_matrix("D", D)
		matrix colnames D  = `: colnames V'

		* 2. CREATE MATRIX B FROM THE USE DATASET
		use "$MyPath/save/USE`world'_`yyyy'", clear
		* A column of B tells me: for an industry, how much is used of each commodity per dollar of industry output
		drop if inlist(code, "V001", "V002", "V003")
		foreach v of varlist V* {
			gen t = sum(`v')
			replace `v' = `v' / t[_N]
			drop t
		}
		mkmat V* if !inlist(code, "VUsed", "VOther", "T006"), matrix(B) rownames(code)

		matrix BD = B * D
		matrix invBD = inv(I(`=colsof(BD)') -  BD)
		* will be used to scale aggregate investment
		if "`world'" != ""{
			egen FU = rowtotal(C I_* G_C G_I* X)
		}
		else{
			egen FU = rowtotal(C I_* G_C G_I*)
		}
		mkmat FU if !inlist(code, "VUsed", "VOther", "T006"), matrix(FU) rownames(code)
		matrix FU = D * invBD * FU


		* create industry level quantities
		use "$MyPath/save/BEAindustries`world'" if year == `yyyy', clear
		* note that I don't keep tfp and intermediate prices since they no longer make sense after capitalizing prof and fin
		keep year code industryorder industrydesc* intermediate T006 va markup labinc* Ic_* VOther VUsed 
		sort industryorder

		* Smetime markup so big that capital share is negative. In these cases, I sset markup so that implied capital income is exactly zero. Note I need to do it only after capitalizing cause that's where intermediate matters
		* Measuring factor income shares at the sectoral level Ákos Valentinyia,b,c,d,∗, Berthold Herrendorf
		gen landinc = 0
		* agriculture
		replace landinc = 0.18 * (VUsed + VOther + T006) if inlist(code, "V111CA", "V113FF")
		* land is around 30% of property value. but it is very long dated (don't depreciate) so in terms of cobb douglas probably smaller, like 0.2 
		replace landinc = 0.24 * (VUsed + VOther + T006) if inlist(code, "V531", "VHS")
		recast double markup
		replace markup = 1 if inlist(code, "V531", "VHS")
		gen double capinc =  (intermediate + VUsed + VOther + T006) / markup - labinc - intermediate - VUsed - VOther - landinc
		replace markup = (intermediate + VUsed + VOther + T006) / (labinc + intermediate + VUsed + VOther + landinc) if capinc <= 0
		drop capinc
		gen Pi = 1 - 1 / markup
		gen USED = VUsed / (intermediate + VUsed + VOther + T006)
		gen IMPORT = VOther / (intermediate + VUsed + VOther + T006)
		gen H = landinc / (intermediate + VUsed + VOther + T006)
		gen K =  1 - Pi - (labinc + intermediate + VUsed + VOther) / (intermediate + VUsed + VOther + T006) - H
		foreach v of varlist labinc*{
			gen L`=subinstr("`v'", "labinc", "", .)' = `v' / (intermediate + VUsed + VOther + T006)
		}
		assert K >= -1e-3

		mkmat Pi K USED IMPORT L_* H, matrix(ΩNF) rownames(code)
		matrix ΩNF_direct = D' * ΩNF

		mkmat Pi K USED IMPORT L_* H, matrix(ΩNF) rownames(code)
		matrix ΩNF_total = invBD' * D' * ΩNF

		mkmat markup, matrix(markup) rownames(code)
		mata {
			markup = st_matrix("markup")  
			ΩNF = st_matrix("ΩNF")  
			ΩNF_cost = ΩNF :* markup
			st_matrix("ΩNF_cost", ΩNF_cost)
			B = st_matrix("B")  
			B_cost = B * diag(markup)
			st_matrix("B_cost", B_cost)
		}
		matrix BD_cost = B_cost * D
		matrix invBD_cost = inv(I(`=colsof(BD)') - BD_cost)
		matrix ΩNF_cost = invBD_cost' * D' * ΩNF_cost
		matrix colnames ΩNF_cost  = `: colnames ΩNF'

		* now do this complicated normalization dance so that the sum of investment done by each industry aggregate to aggregate investment reported in BEA
		* Note that Y_dual = invDB * D * VA should basically equal to the aggregate output of each industry (see TotalRequirementsDerivation.pdf). In reality, very close
		mkmat Ic_equipment Ic_structures Ic_IPP Ic_prof Ic_fin, matrix(Ic) rownames(code)

		* use the nondomestic one because that's the one that should sum up (actually used one, not the non-imported version)
		use "$MyPath/save/USE_`yyyy'", clear
		drop if inlist(code, "V001", "V002", "V003")
		mkmat I_equipment I_structures I_IPP I_prof I_fin if !inlist(code, "VUsed", "VOther", "T006"), matrix(I) rownames(code)
		mkmat G_I_equipment G_I_structures G_I_IPP G_I_prof G_I_fin if !inlist(code, "VUsed", "VOther", "T006"), matrix(G_I) rownames(code)

		mata {
			Ic = st_matrix("Ic")
			I = st_matrix("I")  
			G_I = st_matrix("G_I")  
			FU = st_matrix("FU")  
			i = selectindex(FU :== 0)
			if (colsum(i) > 0){
				FU[i] = 1
			}
			Ic_total = (Ic :/ colsum(Ic) :* (colsum(I) + colsum(G_I))) :/ FU
			st_matrix("Ic_total", Ic_total)
		}
		matrix colnames Ic_total = `: colnames Ic'
		matrix Ic_direct = D' * Ic_total
		matrix Ic_total = invBD' * D' * Ic_total

		/* ok so get back to use dataset */
		use "$MyPath/save/USE_`yyyy'", clear
		drop if inlist(code, "VUsed", "VOther", "V001", "V002", "V003", "T006")
		egen INTERMEDIATE = rowtotal(V*)
		keep code commoditydesc INTERMEDIATE C I_* G_C G_I* X M
		gen commodityorder = _n


		* I have checked and correlation of 90% with original G_p_ii in KLEMS
		* I just need to reconstruct it to account for capitalization of stuff
		svmat ΩNF_direct, names(col)
		svmat priceindices_direct, names(col)
		foreach v of varlist `:colnames ΩNF' {
			rename `v' `v'_direct
		}
		* add investment
		svmat Ic_direct, names(col)
		replace Ic_prof = 0 if missing(Ic_prof)
		replace Ic_fin = 0 if missing(Ic_fin)
		foreach v of varlist Ic_*{
			rename `v' `v'_direct
		}


		* add input expenditures (cost weighted)
		svmat ΩNF_cost, names(col)
		egen temp = rowtotal(K L_nocol L_col USED IMPORT H)
		assert abs(temp - 1) <= 1e-3 if temp != 0
		drop temp
		* Not meaningful to compute
		drop Pi
		foreach v of varlist `:colnames ΩNF_cost'  {
			rename `v' `v'_cost
		}

		* add input expenditures (income weighted)
		svmat ΩNF_total, names(col)	
		egen temp = rowtotal(Pi K H L_nocol L_col USED IMPORT)
		assert abs(temp - 1) <= 1e-3 if temp != 0
		drop temp

		* add investment
		svmat Ic_total, names(col)
		replace Ic_prof = 0 if missing(Ic_prof)
		replace Ic_fin = 0 if missing(Ic_fin)
		gen year = `yyyy'

		if "`world'" != ""{
			* add DIRECT IMPORT EXPENDITURES
			egen FU = rowtotal(C I_* G_C G_I* X)
			gen ratio = M / (INTERMEDIATE + FU)
			foreach v of varlist Pi K* L* USED IMPORT H Ic*{
				replace `v'  = `v' * (1 + ratio)
			}
			replace IMPORT = IMPORT - ratio
			drop ratio
		}

		tempfile temp`yyyy'
		save `temp`yyyy''
	}

	drop _all
	forval yyyy = 1963/2023{
		append using `temp`yyyy''
	}


	save "$MyPath/save/BEAcommodities`world'", replace
}

