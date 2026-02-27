/**
 * Created by Taylor on 9/30/2017.
 */
$(document).ready(function(){

    $(".burger-nav").on("click", function() {
        //$("#nav2 nav ul").toggleClass("open");
        $("#main-menu").toggleClass("open");
    });

});

$(document).ready(function(){

    $("#stats-menu").on("click", function() {
        //$("#sub-menu").toggleClass("open");
        $("#stats-menu ul").toggleClass("open");
    });

});

$(document).ready(function(){

    $("#misc-menu").on("click", function() {
        //$("#sub-menu").toggleClass("open");
        $("#misc-menu ul").toggleClass("open");
    });

});

$(document).ready(function(){

    $("#years-container, .scrolling-startright").each(function () {

        var max = $(this)[0].scrollWidth-$(this).outerWidth();
        $(this).scrollLeft(max);

        if (max<=0) {
            $("#scrollLeft").removeClass("scrollFadeLeft");
            $("#scrollRight").removeClass("scrollFadeRight");
        }

        var maxvalue=10;
        if (maxvalue>max) {
            maxvalue=max;
        }

        $(this).on('scroll', function () {
            var val = $(this).scrollLeft();
            //console.log(val+' '+max+' '+maxvalue);
            if (val < maxvalue) {
                $("#scrollLeft").removeClass("scrollFadeLeft");
            }
            if (val >= maxvalue) {
                $("#scrollLeft").addClass("scrollFadeLeft");
            }

            if (val <= max-maxvalue) {
                $("#scrollRight").addClass("scrollFadeRight");
            }
            if (val > max-maxvalue) {
                $("#scrollRight").removeClass("scrollFadeRight");
            }
            //console.log(val);
        })
    })

});

$(document).ready(function(){

    $(".scrolling-startleft").each(function () {
        $(this).scrollLeft(0);

        //var initval = $(this).scrollLeft();
        var max = $(this)[0].scrollWidth-$(this).outerWidth();

        var maxvalue=10;
        if (maxvalue>max) {
            maxvalue=max;
        }

        $(this).on('scroll', function () {
            var val = $(this).scrollLeft();
            //console.log(val+' '+max);
            if (val>=max-maxvalue) {
                $(this).nextAll("#scrollRight2").removeClass("scrollFadeRight");
            }
            if (val<max-maxvalue) {
                $(this).nextAll("#scrollRight2").addClass("scrollFadeRight");
            }

            if (val > maxvalue) {
                $(this).nextAll("#scrollLeft2").addClass("scrollFadeLeft");
            }
            if (val <= maxvalue) {
                $(this).nextAll("#scrollLeft2").removeClass("scrollFadeLeft");
            }
        })
    })

});
