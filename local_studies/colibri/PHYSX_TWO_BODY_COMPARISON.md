# Actual PhysX two-body drift controls

CPU-prepared source-USD overlays; actual ovphysx0.5.11 GPU TGS runs. No production solver changes. Source frames, drive, geometry, density and materials retained. All other rigid bodies, including Flower, deactivated; only FrameGround/Frame and their authored revolute joint remain. Initial velocities and ether damping zero. Explicit GPU dynamics/TGS,30position+1velocity iterations,120Hz outersteps, stabilization disabled. TGS iterations are a matched temporal budget, not identical Newton microsteps.

Scripts: `prepare_physx_two_body.py`, `benchmark_physx_two_body.py`. Both60s runs completed normally. Mandatory first1/120s physical step precedes tensor bindings;3600subsequent60Hz samples include velocities. This is an accuracy run, not an FPS benchmark.

| Control | Max base displacement from first step | Final displacement |10–60s net XY drift |
|---|---:|---:|---:|
| PhysX default sleep | see saved JSON |41.69µm | zero after resting |
| PhysX sleep threshold0 |62.40µm |41.63µm |0.157µm |
| PhoenX original existing two-body control | different metric origin | — |41.542mm |
| PhoenX total-normal diagnostic | different metric origin | — |1.990mm |

Awake PhysX maximum XY excursion after10s is0.499µm. It retains small oscillatory velocities (final base linear.510mm/s, angular.00222rad/s), so this is bounded jitter, not exact static convergence. Default control has exactly zero recorded velocities by10s, consistent with sleep but no sleep-status tensor is exposed. The explicit awake result rules out sleeping as the sole explanation for bounded drift.

## Physical matching audit and limitations

Runtime FrameGround mass exactly matches saved Newton properties; Frame relative difference about6.5e-7. COM differences≤80nm, inertia relative differences≤1.17e-6. Collider materials are(.5,.5,0) or(0,0,0); both solvers use arithmetic-average friction mixing, grounddefaultmu=.5. Runtime dynamic collider contact offsets are.138889mm, .75mm,1mm as applicable; rest offsets0. Units: USDcentimeters converted toSI for trajectories; inertias multiplied1e-4, localCOM1e-2. Authored angular drive100stageforce-torque/degree converts to.572957795Nm/rad, damping same, target20degrees; maxforce unlimited.

Remaining contact-search mismatch: PhysX static plane autooffset follows its20mm default while Newton sourcebuilder groundfallback is1mm. Dynamic collideroffsets match, but do not call allcontactoffsets matched until a separate explicitly1mm-plane companion is run. Geometry/collision-manifold generation and patch friction are algorithmically different. Source permits default sleep; awake companion changes only sleep threshold. Stabilization is explicitlyoff; source importer default isoff. Source PhysicsBody.cpp338 defines defaultsleep5e-5*toleranceSpeed² (SI .005m²/s² with default10m/s tolerance speed); no runtime sleep tensor confirmation. No damping/sleep workaround is proposed.

## Joint-angle result changes interpretation

Independent SciPy quaternion composition with original localjointframes gives awakePhysX initial18.8855degrees,10s15.7944degrees,60s15.7927degrees. Max anchorerror50.48µm. Thus rest-drive torque at finalpose is+.042073Nm. PhoenX priorquasisteady25.082degrees implies−.050817Nm. They are not the same equilibrium translated alongtheground.

Independent native torque/row audit (`/tmp/colibri_two_body_independent_torque.json`, `/tmp/colibri_two_body_joint_replay_cpu.json`) explains why quasisteadyPhoenX pose is not proof of physical equilibrium: exact localjoint CPU replay matches6native impulses to4.5e-12Ns and closesdrive residual immediately. Subsequent contacts reopen biasedjoint-slot residual+.08406rad/s; copyaverage aggregate+.13221rad/s. Finalrelax lowers aggregate residual to+.04533rad/s, still nonzero. Do not blame jointblock arithmetic; finite coupled convergence and reconciliation remain implicated.

## Bounded next algorithm comparison

First retain original law/masses/SOR1 and close the plane-search control. Then test Kepler’s concrete friction-history defect independently: interior near-cap static loads should not erase anchors merely because previouslambda exceeds98percent ofcone; actualGPUPhysX records breakage when current unconstrainedtrial exceedscone. Use anexplicitbreakbit and preservevalidanchorhistory, withstick/slip/reentryregressions and unchanged law. This targets demonstrated loststaticmemory, not yet proven explanation for allhingeangleerror. Separately compare postcontactjoint residual/physicalaggregateclosure at fixedbudget. A locallyexactjointsolve alone alreadypasses and cannot solve latercontact/copyreopening.

Artifacts: `/tmp/colibri_physx_two_body60.{json,npz,log}`, `/tmp/colibri_physx_two_body_awake60.{json,npz,log}`, `/tmp/colibri_physx_two_body_match_audit.json`; overlays and `.fixture.json` recordfullauthoring. GPUreleased to root/Mencius afterbothterminalruns.

## Matched1mm ground companion (completed)

`/tmp/colibri_physx_two_body_awake_plane1mm60.json/.npz/.log`:3600frames/60s completedexit0. Onlychange versusawakecontrol is staticplanecontactoffset1mm/rest0, matchingNewton groundfallback. Originaldynamiccollideroffsets unchanged. Sleepthreshold0, stabilizationoff, zeroetherdamping,30/1TGS120Hz retained.

Post10–60s netXY drift **0.933359µm**, maximum excursion **1.101434µm**. Hingeangle10s **15.734435deg**,60s **15.728972deg**. Fromfirstphysicalstep, maximum3Dbase displacement94.495µm, final17.179µm. Awakejitter remains (finalspeed1.553mm/s/angular.01257rad/s); notexactstationarity. This closesstaticgroundoffsetcaveat forboundeddriftcomparison, withoutclaimingidenticalmanifolds/contactlaw.

Processwalltime18.63s includesSDKsetup,readbacks,andoutput; no isolatedFPSclaim. GPUreleasedtoKepler immediatelyterminal.
