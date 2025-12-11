/*
 * MetalListFactory.mm
 *
 *  Created for Metal backend
 */

#include "MetalListFactory.h"
#include "../../Utilities/oxDNAException.h"

#include "MetalNoList.h"
#include "MetalSimpleVerletList.h"

MetalBaseList* MetalListFactory::make_list(input_file &inp) {
	char list_type[256];

	if(getInputString(&inp, "Metal_list", list_type, 0) == KEY_NOT_FOUND || !strcmp("verlet", list_type)) {
		return new MetalSimpleVerletList();
	}
	else if(!strcmp("no", list_type)) {
		return new MetalNoList();
	}
	else {
		throw oxDNAException("Metal_list '%s' is not supported", list_type);
	}
}
