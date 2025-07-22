import awkward as ak

def candidate_selection(events , year):
    j_clean = events.Jet[events.Jet.isclean]
    j_soft = j_clean[j_clean.issoft]

    QvG_key = 'btagPNetQvG' if '202' in year else 'particleNetAK4_QvsG'
    bTag_key = 'btagPNetB' if '202' in year else 'particleNetAK4_B'
    j_candidates = j_soft[ak.argsort(getattr(j_soft,QvG_key), axis=1, ascending=False)] #particleNetAK4_QvsG btagPNetQvG
    j_candidates = j_candidates[:, :5] #consider only the first 5
    j_candidates = j_candidates[ak.argsort(getattr(j_candidates,bTag_key), axis=1, ascending=False)]#particleNetAK4_B btagPNetB

    valid_jets = ak.num(j_candidates) >= 4
    j_candidates = ak.mask(j_candidates,valid_jets) # proceed only if we have at least 2 b-jets and 2 non b-jets

    events['j_bcand'] = ak.pad_none(j_candidates[:,:2], 2,axis=1) # two b-jets

    j_candidates = j_candidates[:,2:] #3 non b-jets
    j_candidates = j_candidates[ak.argsort(j_candidates.pt, axis=1, ascending=False)] #pt sort the jets

    jj_i = ak.argcombinations(j_candidates,2,fields=["j1","j2"]) #take dijet combinations
    jj_i = jj_i[(j_candidates[jj_i.j1]-j_candidates[jj_i.j2]).eta<2.0]
    jj_i = jj_i[(j_candidates[jj_i.j1]+ j_candidates[jj_i.j2]).mass<120.0] #dijet cuts
    
    events['j_tt_mask'] =  ak.pad_none(j_candidates[jj_i.j2].pt>20.0, 3, axis=1) # only apply for TTbar chi square
    events['j_nonbcand'] = j_candidates # 3 non b-jets

    events['qq'] = ak.pad_none(j_candidates[jj_i.j1] + j_candidates[jj_i.j2], 3, axis=1)
    events['qq_gen_mass'] = ak.pad_none((j_candidates[jj_i.j1].matched_gen + j_candidates[jj_i.j2].matched_gen).mass, 3, axis=1)
    events['mbb'] = (events.j_bcand[:,0]+events.j_bcand[:,1]).mass

    return events