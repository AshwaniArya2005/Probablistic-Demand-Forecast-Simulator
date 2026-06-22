# Frozen research models: recorded hashes

Checked 2026-09-21. Each file below was written by `ml/sim_rows.py` (`save()`), which records its SHA-256 in `models/manifest.jsonl` (git-ignored, with the weights); every file on disk was hashed again and compared.
**34 of 34 match.** The forecast tables that every reported result reads were produced by these fits. The hashes are checked here and by the serving export (`models.lock.json`, verified by the FastAPI service
at start-up); the research pickles themselves are not re-hashed each time a script runs.

| file | unit | horizon | role | cutoff | code commit | SHA-256 | matches |
|---|---|---:|---|---|---|---|---|
| dev_F2_P10_mean.json | dev_F2 | 10 | mean | 2014-11-23 | `751516b` | `42e60f29b7fc4b7c2092ebd1c55753a3faa53aadcd871ad882dc39c2b163cb0f` | yes |
| dev_F2_P10_quantile.json | dev_F2 | 10 | quantile | 2014-11-23 | `751516b` | `4a598d4b3ea251df1a14b6abd71f8610431bc5e03e786583f7a9fe15ff73c6ef` | yes |
| dev_F4_P10_mean.json | dev_F4 | 10 | mean | 2015-05-31 | `751516b` | `8c0aa57aadf8c34d5a012ffef07dce9c8ff5fc422ebeefe3c8394243da898a2c` | yes |
| dev_F4_P10_quantile.json | dev_F4 | 10 | quantile | 2015-05-31 | `751516b` | `5d0cba0f3de58528e1195d5d7cdd7ce638b1b15471a66100cea39dbca2ea6d22` | yes |
| sim_test_nofp_v0_P10_mean.json | sim_test_nofp_v0 | 10 | mean | 2015-10-25 | `14a450d` | `9d56288d1874a68346ee68624c2b446dfba70a10784f56ffe5e5314c37d716ea` | yes |
| sim_test_nofp_v0_P10_quantile.json | sim_test_nofp_v0 | 10 | quantile | 2015-10-25 | `14a450d` | `28854c803ffea3d8dc16ef0a4bbe5c8835246a802edd1952f4be2504584af05b` | yes |
| sim_test_nofp_v1_P10_mean.json | sim_test_nofp_v1 | 10 | mean | 2015-11-22 | `14a450d` | `e31097f54e5a0692c95ac3039b040c634bd142a5c3b39c12729e9f9004a0df04` | yes |
| sim_test_nofp_v1_P10_quantile.json | sim_test_nofp_v1 | 10 | quantile | 2015-11-22 | `14a450d` | `1077c9a460df06fd29887570c4b122d337a9f5b422af49f5d85b50c6fc595441` | yes |
| sim_test_nofp_v2_P10_mean.json | sim_test_nofp_v2 | 10 | mean | 2016-01-17 | `14a450d` | `c3fa52aef1718d09c8445168c57a95e8b7f48108c2804942dd4eaa8807ce571e` | yes |
| sim_test_nofp_v2_P10_quantile.json | sim_test_nofp_v2 | 10 | quantile | 2016-01-17 | `14a450d` | `fe660bd9cf42cd67aec0807131040c861525145d9f5b94c1e7cf5ef01ddf87b8` | yes |
| sim_test_nofp_v3_P10_mean.json | sim_test_nofp_v3 | 10 | mean | 2016-03-13 | `14a450d` | `7f8a3b39050028d58f638cd13d8f365b7b2ebcc385da4d6ca1a71f75ed43531f` | yes |
| sim_test_nofp_v3_P10_quantile.json | sim_test_nofp_v3 | 10 | quantile | 2016-03-13 | `14a450d` | `67c872314f4cb827de4945d69a55b5fc621279406ee64a2a000c261f0d354c9e` | yes |
| sim_test_nofp_v4_P10_mean.json | sim_test_nofp_v4 | 10 | mean | 2016-05-08 | `14a450d` | `1c9668cdf528e5491e61702faaaf5265551bfa8d5107cfcf41f99260f9026331` | yes |
| sim_test_nofp_v4_P10_quantile.json | sim_test_nofp_v4 | 10 | quantile | 2016-05-08 | `14a450d` | `1b638587f9850403a070be1f1fee2e7a1c32364cd89e89df0992dcad29a19316` | yes |
| test_v0_P10_mean.json | test_v0 | 10 | mean | 2015-10-25 | `1dec014` | `1b9845c85ebded75391024fb84aa75f05898c71059e08cc3d679de0f9bc908fa` | yes |
| test_v0_P10_quantile.json | test_v0 | 10 | quantile | 2015-10-25 | `1dec014` | `177a168a9033ae20a1ba28b2a753fbcb73f8f107de7dab2fbbc0caf9eb6fb60d` | yes |
| test_v0_P14_mean.json | test_v0 | 14 | mean | 2015-10-25 | `14a450d` | `d34cb842eebf13331cba7d8533a0a7416dfe3d7d501682b4a2bd61550951cd0d` | yes |
| test_v0_P14_quantile.json | test_v0 | 14 | quantile | 2015-10-25 | `14a450d` | `430494c65b029756ec2919e4e956b62221b7e1a1be53e602b7f26b034bb8e345` | yes |
| test_v1_P10_mean.json | test_v1 | 10 | mean | 2015-11-22 | `1dec014` | `1aaf29ec62190139be58824682263872d1ca5e51f7d0016bc1dc9be851fe2591` | yes |
| test_v1_P10_quantile.json | test_v1 | 10 | quantile | 2015-11-22 | `1dec014` | `9bf8d7572e7fb863cd6d29026da58b9914eee1aede92cd91a3b27ed648d4ae18` | yes |
| test_v1_P14_mean.json | test_v1 | 14 | mean | 2015-11-22 | `14a450d` | `6032b2a8f800de590e32157a8804eac1280740ef14c8a28f561835eedeb7c5e9` | yes |
| test_v1_P14_quantile.json | test_v1 | 14 | quantile | 2015-11-22 | `14a450d` | `78a4d5dc90c704149852eef8460fba427a5a47728235163b2803f73a0afd94c9` | yes |
| test_v2_P10_mean.json | test_v2 | 10 | mean | 2016-01-17 | `1dec014` | `aebc4c1789b0a5df993c1899f7f612f9ed18ba8ab043653838018b11e1389289` | yes |
| test_v2_P10_quantile.json | test_v2 | 10 | quantile | 2016-01-17 | `1dec014` | `c319ea3b2281dbe2c5aa9ac3b6b294d723f7eaf216541ed50046a5b28df00e6b` | yes |
| test_v2_P14_mean.json | test_v2 | 14 | mean | 2016-01-17 | `14a450d` | `25448ae7be01aef7551e3dd16ac7faa50e85d151ab51f1e20f087d30ded15fd4` | yes |
| test_v2_P14_quantile.json | test_v2 | 14 | quantile | 2016-01-17 | `14a450d` | `2edbe4308f1d579859b5dc04d07694c69cbc621433aa72513ce916956b3984ea` | yes |
| test_v3_P10_mean.json | test_v3 | 10 | mean | 2016-03-13 | `1dec014` | `76fe04101a6e18e773353e8de5d1743e3b097a3867668039ffae24056af122c3` | yes |
| test_v3_P10_quantile.json | test_v3 | 10 | quantile | 2016-03-13 | `1dec014` | `3b78e5291f65f1693fb23162a2a81adc29d95e40bdd5f0d4901d7d431330a5d3` | yes |
| test_v3_P14_mean.json | test_v3 | 14 | mean | 2016-03-13 | `14a450d` | `a760693933b9550691c24534e3e5ec039ced2e03a1eb8254493dba43a9c7520f` | yes |
| test_v3_P14_quantile.json | test_v3 | 14 | quantile | 2016-03-13 | `14a450d` | `00634787aea4665801cef1f10f57eea7b370fa0597a1f30a7d7a0cb8657a6bdf` | yes |
| test_v4_P10_mean.json | test_v4 | 10 | mean | 2016-05-08 | `1dec014` | `da7e9d2420c6c7525db8f9ab118405dd87f0198ffe8bfe2b4c4f84ec4030d2d4` | yes |
| test_v4_P10_quantile.json | test_v4 | 10 | quantile | 2016-05-08 | `1dec014` | `75edec266441d96481c4b802f10ec38a50053c45ac202d5e29d395046e94bf2b` | yes |
| test_v4_P14_mean.json | test_v4 | 14 | mean | 2016-05-08 | `14a450d` | `1e1dad90f4be9f74527a9666ad57195d64475510badf6c74251a5a6580dc8805` | yes |
| test_v4_P14_quantile.json | test_v4 | 14 | quantile | 2016-05-08 | `14a450d` | `e4a12e2793cf3ca606fa6f9ebcc38ff0c4e09afe81a69ac278bb2a894be1c6bb` | yes |
