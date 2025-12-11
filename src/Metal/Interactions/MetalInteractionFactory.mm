/*
 * MetalInteractionFactory.mm
 *
 *  Created for Metal backend
 */

#include "MetalInteractionFactory.h"
#include "../../Utilities/oxDNAException.h"

#include "MetalDNAInteraction.h"

MetalBaseInteraction* MetalInteractionFactory::make_interaction(input_file &inp) {
	std::string interaction_type;
	if(getInputString(&inp, "interaction_type", interaction_type, 1) == KEY_NOT_FOUND) {
		throw oxDNAException("Cannot search for interaction_type in input_file");
	}

	if(interaction_type.compare("DNA") == 0 || interaction_type.compare("DNA2") == 0) {
		return new MetalDNAInteraction();
	}
	else {
		throw oxDNAException("Metal interaction '%s' is not supported yet", interaction_type.c_str());
	}
}
