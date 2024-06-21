
(function( $ ) {

	$.fn.jqLoad = function( parameter ) {

		var epd;
		var rightNow = new Date().getTime();
		
		// if parameter is empty, look for data-expire attributes
				
		if ( arguments.length == 0 ) {
			$( "*[data-load]" ).each( function() {
				epd = new Date( $( this ).data( 'load' )*1000 );
				console.log(epd)
				
				if ( epd < rightNow ) {
					$( this ).hide();
				}
			});
		} 
				
		// if parameter type is an array, then loop through it individually
		
		else if (typeof parameter == 'array') {
			console.log("This feature is yet to be implemented!")
		}
		
		// if parameter type appears to be a date/time stamp, check that
		// against each matched element.

		else if (typeof parameter == 'string') {
						
			expiryDate = new Date( parameter ).getTime();
						
			if (expiryDate < rightNow) {
				this.each( function() {
					$( this ).hide();
				} );	
			}
		}

	};

}( jQuery ));