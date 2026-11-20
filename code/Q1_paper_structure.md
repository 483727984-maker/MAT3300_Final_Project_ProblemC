# Question 1: recommended paper structure

## 1. Network definition and data handling

Define a directed graph (G=(V,E)), where each artist is a node and (i\to j) means the data report artist (i) as an influence on artist (j). Use every valid directed pair once, with baseline edge weight (w_{ij}=1), because the file records whether a relation is reported but not its strength. Store genre and active-start decade as node attributes.

State that active-start decade is an approximate artist attribute, not the date a particular influence occurred. Keep the 733 reversed decade-order records in the primary network. Show their count and assess their effect in a separate sensitivity comparison.

## 2. Influence measures

Use three measures with distinct interpretations:

1. **Out-degree** (d_i^{out}=\sum_j A_{ij}): number of directly named followers.
2. **Outgoing Katz centrality**
   \[
   K_i=\left[(I-\alpha A)^{-1}\mathbf 1\right]_i
      =1+\alpha(A\mathbf 1)_i+\alpha^2(A^2\mathbf 1)_i+\cdots,
   \]
   which counts downstream paths with geometric attenuation. Report the selected \(\alpha\), its convergence bound, and sensitivity to nearby values.
3. **PageRank on the reversed graph** as a comparison measure. State explicitly that reversing edges makes rank flow from reported followers back to their influencers.

Report the measures separately. Compare rank correlations and top-​(k) overlap; avoid forming an unmotivated weighted composite. Explain that high correlation overall can coexist with different top-ranked artists.

## 3. Whole-network results

Report node and edge counts, density, weak/strong component summaries, zero in/out-degree counts, and the distribution of the active-start decade gap. Interpret zero out-degree as having no recorded followers in this dataset, not as proof of no influence.

## 4. Dynamic view using follower cohorts

Group records by the follower's active-start decade and recompute Katz within each cohort graph. Call these **follower-start-decade cohorts**. They describe which influences are reported by artists who began in each decade; they do not date when the influence happened. Discuss recurring versus cohort-specific influencers, and note the smaller sample in the latest decade.

## 5. Jazz subnetwork case study

Define the Jazz-induced subnetwork by retaining edges whose two endpoints are both labeled Jazz. Report its size and topology, rank artists using within-Jazz Katz and out-degree, and separately count Jazz-to-Pop/Rock edges to show cross-genre outflow. Do not interpret within-subnetwork scores as directly comparable in magnitude to whole-network scores; compare ranks or roles instead.

## 6. Robustness and limitations

Include (a) Katz attenuation sensitivity, (b) all edges versus the optional active-start-order filtered graph, and (c) differences between Katz, out-degree, and reverse PageRank. Emphasize that influence edges are reported relationships, not causal proof, and that the network is unweighted.

## Figure plan

| Figure | Content | Purpose |
|---|---|---|
| 1 | Workflow diagram: data → directed network → indicators → full network/cohorts/subnetwork | Orient the reader before equations and results |
| 2 | Top artists by Katz alongside Katz-versus-out-degree scatter | Show leading artists and what indirect reach adds |
| 3 | Two-panel sensitivity plot: Katz \(\alpha\) stability; all-edge versus filtered scores | Demonstrate parameter and time-filter robustness |
| 4 | Cohort heatmap of recurring top-10 artists and their rank by decade | Show dynamic influencers without implying exact influence dates |
| 5 | Focused directed Jazz ego-subnetwork plus within-Jazz Katz ranking | Show interpretable paths while keeping the network legible |

Keep full-network topology statistics in a compact table rather than a hairball plot. Every plotted subnetwork should state its selection rule, and every node-size/color encoding should be defined in the caption.
