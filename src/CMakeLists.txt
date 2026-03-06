set(CMAKE_POSITION_INDEPENDENT_CODE ON)

SET(lib_name oxdna_common)

IF(CMAKE_BUILD_TYPE MATCHES Debug)
	SET(exe_name oxDNA_debug)
ELSEIF(MPI)
	SET(exe_name oxDNA_mpi)
ELSE()
	SET(exe_name oxDNA)
ENDIF()

SET(observables_SOURCES
	Observables/BaseObservable.cpp
	Observables/ObservableFactory.cpp
	Observables/ObservableOutput.cpp
	Observables/Step.cpp
	Observables/PotentialEnergy.cpp
	Observables/StackingPropensity.cpp
	Observables/CrossStackingPropensity.cpp
	Observables/KineticEnergy.cpp
	Observables/TotalEnergy.cpp
	Observables/BackendInfo.cpp	
	Observables/HBEnergy.cpp
	Observables/OrderParameterValues.cpp
	Observables/HBList.cpp
	Observables/StrandwiseBonds.cpp
	Observables/PairEnergy.cpp
	Observables/PairForce.cpp
	Observables/ForceEnergy.cpp
	Observables/Pressure.cpp
	Observables/StressAutocorrelation.cpp
	Observables/Density.cpp
	Observables/Rdf.cpp
	Observables/Distance.cpp
	Observables/DensityProfile.cpp
	Observables/ParticlePosition.cpp
	Observables/CoaxVariables.cpp
	Observables/Pitch.cpp
	Observables/SaltExtrapolation.cpp
	Observables/ExternalTorque.cpp
	Observables/MeanVectorCosine.cpp
	Observables/VectorAngle.cpp
	Observables/Checkpoint.cpp
	Observables/Contacts.cpp
	Observables/Writhe.cpp
	Observables/UnstackedList.cpp
	Observables/PlectonemePosition.cpp
	Observables/StretchedBonds.cpp
	Observables/StructureFactor.cpp
	Observables/FormFactor.cpp
	Observables/UnstackedList.cpp
	Observables/PlectonemePosition.cpp
	Observables/TEPPlectonemePosition.cpp
	Observables/AverageEnergy.cpp
	Observables/ContactMap.cpp
	Observables/AllVectors.cpp
	Observables/ExternalForce.cpp
	Observables/ParticleForceAndTorque.cpp
	Observables/Coordination.cpp
	Observables/Configurations/Configuration.cpp
	Observables/Configurations/BinaryConfiguration.cpp
	Observables/Configurations/TclOutput.cpp
	Observables/Configurations/PdbOutput.cpp
	Observables/Configurations/ChimeraOutput.cpp
	Observables/Configurations/TEPtclOutput.cpp
	Observables/Configurations/TEPxyzOutput.cpp
	Observables/Configurations/JordanOutput.cpp
)

SET(backends_SOURCES
	Backends/BackendFactory.cpp
	Backends/SimBackend.cpp
	Backends/AnalysisBackend.cpp 
	Backends/MDBackend.cpp
	Backends/MCBackend.cpp
	Backends/MD_CPUBackend.cpp 
	Backends/MC_CPUBackend.cpp
	Backends/MC_CPUBackend2.cpp
	Backends/VMMC_CPUBackend.cpp
	Backends/MinBackend.cpp
	Backends/FIREBackend.cpp
	Backends/FIREBackend.h
	Backends/FFS_MD_CPUBackend.cpp
	Backends/Thermostats/SRDThermostat.cpp
	Backends/Thermostats/DPDThermostat.cpp
	Backends/Thermostats/ThermostatFactory.cpp
	Backends/Thermostats/BaseThermostat.cpp
	Backends/Thermostats/NoThermostat.cpp
	Backends/Thermostats/BrownianThermostat.cpp
	Backends/Thermostats/RefreshThermostat.cpp
	Backends/Thermostats/BussiThermostat.cpp
	Backends/Thermostats/LangevinThermostat.cpp
)

SET(lists_SOURCES
	Lists/BaseList.cpp
	Lists/NoList.cpp
	Lists/Cells.cpp
	Lists/RodCells.cpp
	Lists/VerletList.cpp
	Lists/BinVerletList.cpp
	Lists/ListFactory.cpp
)

SET(interactions_SOURCES
	Interactions/InteractionFactory.cpp
	Interactions/InteractionUtils.cpp
	Interactions/Mesh.cpp
	Interactions/BaseInteraction.cpp
	Interactions/LJInteraction.cpp
	Interactions/DNAInteraction.cpp
	Interactions/DNA2Interaction.cpp
	Interactions/DNA2SDInteraction.cpp
	Interactions/DNA3Interaction.cpp
	Interactions/DNAInteraction_relax.cpp
	Interactions/RNAInteraction.cpp
	Interactions/RNAInteraction2.cpp
	Interactions/RNAInteraction_relax.cpp
	Interactions/JordanInteraction.cpp
	Interactions/TEPInteraction.cpp
	Interactions/HSInteraction.cpp
	Interactions/DHSInteraction.cpp
	Interactions/HardCylinderInteraction.cpp
	Interactions/HardSpheroCylinderInteraction.cpp
	Interactions/BoxInteraction.cpp
	Interactions/PatchyInteraction.cpp
	Interactions/PatchyInteractionDan.cpp
	Interactions/KFInteraction.cpp
	Interactions/CustomInteraction.cpp
	Interactions/DRHInteraction.cpp
	Interactions/DRHInteraction_relax.cpp
	Interactions/ANMInteraction.cpp
	Interactions/DNANMInteraction.cpp
	Interactions/RNANMInteraction.cpp
)

SET(move_SOURCES
	Backends/MCMoves/BaseMove.cpp
	Backends/MCMoves/MoveFactory.cpp
	Backends/MCMoves/MCTras.cpp
	Backends/MCMoves/MCRot.cpp
	Backends/MCMoves/Pivot.cpp
	Backends/MCMoves/VMMC.cpp
	Backends/MCMoves/VolumeMove.cpp
	Backends/MCMoves/MoleculeVolumeMove.cpp
	Backends/MCMoves/ShapeMove.cpp
	Backends/MCMoves/RotateSite.cpp
)

SET(forces_SOURCES
	Forces/ForceFactory.cpp
	Forces/BaseForce.cpp
	Forces/COMForce.cpp
	Forces/ConstantRateForce.cpp
	Forces/SawtoothForce.cpp
	Forces/ConstantRateTorque.cpp
	Forces/ConstantTrap.cpp
	Forces/MovingTrap.cpp
	Forces/LowdimMovingTrap.cpp
	Forces/MutualTrap.cpp
	Forces/RepulsiveEllipsoid.cpp
	Forces/RepulsionPlane.cpp
	Forces/RepulsionPlaneMoving.cpp
	Forces/RepulsiveSphere.cpp
	Forces/RepulsiveSphereSmooth.cpp
	Forces/LJWall.cpp
	Forces/HardWall.cpp
	Forces/AlignmentField.cpp
	Forces/GenericCentralForce.cpp
	Forces/LJCone.cpp
	Forces/YukawaSphere.cpp
	Forces/Metadynamics/LTCoordination.cpp
	Forces/Metadynamics/LTCOMTrap.cpp
	Forces/Metadynamics/LTCOMAngleTrap.cpp
	Forces/Metadynamics/LT2DCOMTrap.cpp
	Forces/Metadynamics/LTAtanCOMTrap.cpp
	Forces/Metadynamics/meta_utils.cpp
	Forces/AttractionPlane.cpp
	Forces/RepulsiveKeplerPoinsot.cpp
	Forces/RepulsiveSphereMoving.cpp

)

SET(box_SOURCES
	Boxes/BaseBox.cpp
	Boxes/CubicBox.cpp
	Boxes/OrthogonalBox.cpp
	Boxes/LeesEdwardsCubicBox.cpp
	Boxes/BoxFactory.cpp
)

SET(common_SOURCES
	Particles/JordanParticle.cpp
	Particles/TEPParticle.cpp
	Particles/RNANucleotide.cpp
	Particles/BaseParticle.cpp
	Particles/DNANucleotide.cpp
	Particles/PatchyParticle.cpp
	Particles/PatchyParticleDan.cpp
	Particles/SpheroCylinder.cpp
	Particles/CustomParticle.cpp
	Particles/Molecule.cpp
	Particles/ANMParticle.cpp
	Managers/SimManager.cpp
	Utilities/OrderParameters.cpp
	Utilities/Weights.cpp
	Utilities/Histogram.cpp
	Utilities/LR_vector.cpp
	Utilities/LR_matrix.cpp
	Utilities/Utils.cpp
	Utilities/oxDNAException.cpp
	Utilities/Logger.cpp
	Utilities/parse_input/parse_input.cpp
	Utilities/time_scales/time_scales.cpp
	Utilities/SignalManager.cpp
	Utilities/ConfigInfo.cpp
	Utilities/FlattenedConfigInfo.cpp
	Utilities/TopologyParser.cpp
	PluginManagement/PluginManager.cpp
	${forces_SOURCES}
	${observables_SOURCES}
	${lists_SOURCES}
	${backends_SOURCES}
	${interactions_SOURCES}
	${move_SOURCES}
	${box_SOURCES}
)

SET(DNAnalysis_SOURCES
	DNAnalysis.cpp
	Managers/AnalysisManager.cpp
)

SET(confGenerator_SOURCES
	confGenerator.cpp
	Managers/GeneratorManager.cpp
)

IF(MPI)
	FIND_PACKAGE(MPI REQUIRED)
	INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
	
	LIST(APPEND common_SOURCES
		Backends/PT_VMMC_CPUBackend.cpp
		Managers/ParallelManager.cpp
	)
	
	SET(oxDNA_SOURCES
		oxDNA_mpi.cpp 
	)
ELSE()
	SET(oxDNA_SOURCES
		oxDNA.cpp 
	)
ENDIF()
	
IF(CUDA)
	CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
	
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")
	ADD_DEFINITIONS(-DGPU_BACKEND_CUDA)
	
	LIST(APPEND common_SOURCES
		CUDA/CUDA_sort.cu
		CUDA/CUDAUtils.cu
		CUDA/cuda_utils/cuda_device_utils.cu
		CUDA/Backends/CUDABaseBackend.cu
		CUDA/Backends/MD_CUDABackend.cu
		CUDA/Interactions/CUDABaseInteraction.cu
		CUDA/Interactions/CUDADNAInteraction.cu
		CUDA/Interactions/CUDADNA3Interaction.cu
		CUDA/Interactions/CUDALJInteraction.cu
		CUDA/Interactions/CUDARNAInteraction.cu
		CUDA/Interactions/CUDAPatchyInteraction.cu
		CUDA/Interactions/CUDATEPInteraction.cu
		CUDA/Interactions/CUDAInteractionFactory.cu
		CUDA/Interactions/CUDADNANMInteraction.cu
		CUDA/Lists/CUDAListFactory.cu
		CUDA/Lists/CUDANoList.cu
		CUDA/Lists/CUDASimpleVerletList.cu
		CUDA/Lists/CUDABinVerletList.cu
		CUDA/Thermostats/CUDABaseThermostat.cu
		CUDA/Thermostats/CUDANoThermostat.cu
		CUDA/Thermostats/CUDAThermostatFactory.cu
		CUDA/Thermostats/CUDABrownianThermostat.cu
		CUDA/Thermostats/CUDASRDThermostat.cu
		CUDA/Thermostats/CUDALangevinThermostat.cu
		CUDA/Thermostats/CUDABussiThermostat.cu
		CUDA/Thermostats/CUDARefreshThermostat.cu
		Utilities/Timings.cpp
	)
	
	IF(CUDA_DOUBLE)
		ADD_DEFINITIONS(-DCUDA_DOUBLE_PRECISION)
	ELSE()
		LIST(APPEND common_SOURCES
			CUDA/Backends/MD_CUDAMixedBackend.cu 
			CUDA/Backends/FFS_MD_CUDAMixedBackend.cu
		)
	ENDIF()
	
	SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")

	set(CUDA_COMMON_FLAGS
		-use_fast_math
		--compiler-options=-ffast-math
		--compiler-options=-O3
		-Xcudafe=--diag_suppress=unrecognized_gcc_pragma
	)
	if(G)
		list(APPEND CUDA_COMMON_FLAGS -lineinfo)
	endif()

	if(DEBUG)
		list(APPEND CUDA_COMMON_FLAGS -lineinfo -G)
	endif()

	if(CMAKE_VERSION VERSION_LESS 3.18)
		# we need the library to be shared or the oxpy compilation fails
		CUDA_ADD_LIBRARY(${lib_name} SHARED ${common_SOURCES})
		CUDA_ADD_EXECUTABLE(${exe_name} ${oxDNA_SOURCES})
		CUDA_ADD_EXECUTABLE(DNAnalysis ${DNAnalysis_SOURCES} Utilities/Timings.cpp)
	else()
		add_library(${lib_name} SHARED ${common_SOURCES})
		target_link_libraries(${lib_name} PRIVATE CUDA::cudart)
		target_compile_options(${lib_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMMON_FLAGS}>)

		add_executable(${exe_name} ${oxDNA_SOURCES})
		target_link_libraries(${exe_name} PRIVATE CUDA::cudart)
		target_compile_options(${exe_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMMON_FLAGS}>)

		add_executable(DNAnalysis ${DNAnalysis_SOURCES}	Utilities/Timings.cpp)
		target_link_libraries(DNAnalysis PRIVATE CUDA::cudart)
		target_compile_options(DNAnalysis PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMMON_FLAGS}>)
	endif()

	MESSAGE(STATUS "${CUDA_COMMON_FLAGS}")
ELSEIF(ROCM)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")
	ADD_DEFINITIONS(-DGPU_BACKEND_ROCM)
	get_filename_component(HIP_BIN_DIR "${CMAKE_HIP_COMPILER}" DIRECTORY)
	get_filename_component(HIP_LLVM_DIR "${HIP_BIN_DIR}" DIRECTORY)
	get_filename_component(ROCM_ROOT_DIR "${HIP_LLVM_DIR}" DIRECTORY)
	find_library(ROCM_HIP_RUNTIME amdhip64
		HINTS
			${ROCM_ROOT_DIR}/lib
			/opt/rocm/lib
		REQUIRED
	)

	LIST(APPEND common_SOURCES
		ROCM/CUDA_sort.cu
		ROCM/CUDAUtils.cu
		ROCM/cuda_utils/cuda_device_utils.cu
		ROCM/Backends/CUDABaseBackend.cu
		ROCM/Backends/MD_CUDABackend.cu
		ROCM/Interactions/CUDABaseInteraction.cu
		ROCM/Interactions/CUDADNAInteraction.cu
		ROCM/Interactions/CUDADNA3Interaction.cu
		ROCM/Interactions/CUDALJInteraction.cu
		ROCM/Interactions/CUDARNAInteraction.cu
		ROCM/Interactions/CUDAPatchyInteraction.cu
		ROCM/Interactions/CUDATEPInteraction.cu
		ROCM/Interactions/CUDAInteractionFactory.cu
		ROCM/Interactions/CUDADNANMInteraction.cu
		ROCM/Lists/CUDAListFactory.cu
		ROCM/Lists/CUDANoList.cu
		ROCM/Lists/CUDASimpleVerletList.cu
		ROCM/Lists/CUDABinVerletList.cu
		ROCM/Thermostats/CUDABaseThermostat.cu
		ROCM/Thermostats/CUDANoThermostat.cu
		ROCM/Thermostats/CUDAThermostatFactory.cu
		ROCM/Thermostats/CUDABrownianThermostat.cu
		ROCM/Thermostats/CUDALangevinThermostat.cu
		Utilities/Timings.cpp
	)

	IF(CUDA_DOUBLE)
		ADD_DEFINITIONS(-DCUDA_DOUBLE_PRECISION)
	ELSE()
		LIST(APPEND common_SOURCES
			ROCM/Backends/MD_CUDAMixedBackend.cu
			ROCM/Backends/FFS_MD_CUDAMixedBackend.cu
		)
	ENDIF()

	set_source_files_properties(
		ROCM/CUDA_sort.cu
		ROCM/CUDAUtils.cu
		ROCM/cuda_utils/cuda_device_utils.cu
		ROCM/Backends/CUDABaseBackend.cu
		ROCM/Backends/MD_CUDABackend.cu
		ROCM/Interactions/CUDABaseInteraction.cu
		ROCM/Interactions/CUDADNAInteraction.cu
		ROCM/Interactions/CUDADNA3Interaction.cu
		ROCM/Interactions/CUDALJInteraction.cu
		ROCM/Interactions/CUDARNAInteraction.cu
		ROCM/Interactions/CUDAPatchyInteraction.cu
		ROCM/Interactions/CUDATEPInteraction.cu
		ROCM/Interactions/CUDAInteractionFactory.cu
		ROCM/Interactions/CUDADNANMInteraction.cu
		ROCM/Lists/CUDAListFactory.cu
		ROCM/Lists/CUDANoList.cu
		ROCM/Lists/CUDASimpleVerletList.cu
		ROCM/Lists/CUDABinVerletList.cu
		ROCM/Thermostats/CUDABaseThermostat.cu
		ROCM/Thermostats/CUDANoThermostat.cu
		ROCM/Thermostats/CUDAThermostatFactory.cu
		ROCM/Thermostats/CUDABrownianThermostat.cu
		ROCM/Thermostats/CUDALangevinThermostat.cu
		ROCM/Backends/MD_CUDAMixedBackend.cu
		ROCM/Backends/FFS_MD_CUDAMixedBackend.cu
		PROPERTIES LANGUAGE HIP
	)

	set(ROCM_COMMON_FLAGS
		-ffast-math
		-O3
	)

	add_library(${lib_name} SHARED ${common_SOURCES})
	target_include_directories(${lib_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ROCM ${ROCM_ROOT_DIR}/include)
	target_compile_definitions(${lib_name} PRIVATE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP __HIP_PLATFORM_AMD__= __HIP_PLATFORM_HCC__=)
	target_compile_options(${lib_name} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:${ROCM_COMMON_FLAGS}>)
	target_link_libraries(${lib_name} PRIVATE ${ROCM_HIP_RUNTIME})
	set_target_properties(${lib_name} PROPERTIES BUILD_RPATH "${ROCM_ROOT_DIR}/lib")

	add_executable(${exe_name} ${oxDNA_SOURCES})
	target_compile_definitions(${exe_name} PRIVATE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP __HIP_PLATFORM_AMD__= __HIP_PLATFORM_HCC__=)
	target_compile_options(${exe_name} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:${ROCM_COMMON_FLAGS}>)
	target_link_libraries(${exe_name} PRIVATE ${ROCM_HIP_RUNTIME})
	target_include_directories(${exe_name} PRIVATE ${ROCM_ROOT_DIR}/include)
	set_target_properties(${exe_name} PROPERTIES BUILD_RPATH "${ROCM_ROOT_DIR}/lib")

	add_executable(DNAnalysis ${DNAnalysis_SOURCES} Utilities/Timings.cpp)
	target_compile_definitions(DNAnalysis PRIVATE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP __HIP_PLATFORM_AMD__= __HIP_PLATFORM_HCC__=)
	target_compile_options(DNAnalysis PRIVATE $<$<COMPILE_LANGUAGE:HIP>:${ROCM_COMMON_FLAGS}>)
	target_link_libraries(DNAnalysis PRIVATE ${ROCM_HIP_RUNTIME})
	target_include_directories(DNAnalysis PRIVATE ${ROCM_ROOT_DIR}/include)
	set_target_properties(DNAnalysis PROPERTIES BUILD_RPATH "${ROCM_ROOT_DIR}/lib")
ELSE()
	ADD_DEFINITIONS(-DNO_GPU_BACKEND)
	SET(common_SOURCES
		${common_SOURCES}
		Utilities/Timings.cpp
	)

	ADD_EXECUTABLE(${exe_name} ${oxDNA_SOURCES})
	ADD_EXECUTABLE(DNAnalysis ${DNAnalysis_SOURCES})
	ADD_LIBRARY(${lib_name} ${common_SOURCES})
ENDIF(CUDA)

ADD_EXECUTABLE(confGenerator ${confGenerator_SOURCES})
IF(ROCM)
	target_compile_definitions(confGenerator PRIVATE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP __HIP_PLATFORM_AMD__= __HIP_PLATFORM_HCC__=)
	target_link_libraries(confGenerator PRIVATE ${ROCM_HIP_RUNTIME})
	target_include_directories(confGenerator PRIVATE ${ROCM_ROOT_DIR}/include)
	set_target_properties(confGenerator PROPERTIES BUILD_RPATH "${ROCM_ROOT_DIR}/lib")
ENDIF()

IF(MPI)
	target_link_libraries(${lib_name} PRIVATE ${CMAKE_DL_LIBS} ${MPI_LIBRARIES})
ELSE()
	target_link_libraries(${lib_name} PRIVATE ${CMAKE_DL_LIBS})
ENDIF()

target_link_libraries(${exe_name} PRIVATE ${lib_name})
target_link_libraries(DNAnalysis PRIVATE ${lib_name})
target_link_libraries(confGenerator PRIVATE ${lib_name})

# we add these executable as dependencies for the test targets
ADD_DEPENDENCIES(test_run ${exe_name} DNAnalysis confGenerator)
ADD_DEPENDENCIES(test_quick ${exe_name} DNAnalysis confGenerator)
IF(TARGET test_rocm_run)
	ADD_DEPENDENCIES(test_rocm_run ${exe_name} DNAnalysis confGenerator)
ENDIF()
IF(TARGET test_rocm_quick)
	ADD_DEPENDENCIES(test_rocm_quick ${exe_name} DNAnalysis confGenerator)
ENDIF()
