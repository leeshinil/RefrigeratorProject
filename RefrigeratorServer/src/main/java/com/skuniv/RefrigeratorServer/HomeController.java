package com.skuniv.RefrigeratorServer;

import java.net.URLEncoder;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;
import java.text.DateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Locale;

import javax.servlet.http.HttpServletRequest;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import model.RefItem;

/**
 * Handles requests for the application home page.
 */
@Controller
public class HomeController {

	private static final Logger logger = LoggerFactory.getLogger(HomeController.class);

	/**
	 * Simply selects the home view to render by returning its name.
	 */
	@RequestMapping(value = "/", method = RequestMethod.GET)
	public String home(Locale locale, Model model) {
		logger.info("Welcome home! The client locale is {}.", locale);

		Date date = new Date();
		DateFormat dateFormat = DateFormat.getDateTimeInstance(DateFormat.LONG, DateFormat.LONG, locale);

		String formattedDate = dateFormat.format(date);

		model.addAttribute("serverTime", formattedDate);

		return "home";
	}

	////////////// 식재료 불러오기 ///////////////////////
	@RequestMapping(value = "/getRefItems", method = RequestMethod.POST, headers = "Content-Type=application/x-www-form-urlencoded")
	public @ResponseBody String getRefItems(HttpServletRequest request, Model model) throws Exception {

		Connection conn = new DBconn.DBconnection().DBconn();
		ArrayList<RefItem> refItemList = new ArrayList<RefItem>();

		try {
			Statement st = null;
			Statement ddlst = null;
			ResultSet rs = null; // 결과값을 받아오기 위한 변수
			int r;

			ddlst = conn.createStatement(); // DDL쿼리처리를 위한 변수
			st = conn.createStatement(); // SELECT를 위한 변수

			rs = st.executeQuery("select * from refrigerator");

			while (rs.next()) {
				RefItem refItem = new RefItem();
				refItem.setIndex(rs.getInt(1));
				refItem.setName(rs.getString(2));
				refItem.setInputDate(rs.getString(3));
				refItem.setExpiration(rs.getString(4));
				
		
				refItemList.add(refItem);
			}

			System.out.println("success get RefItem!");

			rs.close();
			st.close();
			ddlst.close();
			conn.close();

		} catch (Exception e) {
			System.out.println("failed connect");
			e.printStackTrace();
		}

		Gson gson = new GsonBuilder().disableHtmlEscaping().create();

		return gson.toJson(refItemList);
	}

	////////////////// 식재료 삭제하기 ////////////////////////////
	@RequestMapping(value = "/deleteRefItemInfo", method = RequestMethod.POST, headers = "Content-Type=application/x-www-form-urlencoded")
	public @ResponseBody void deleteRefItemInfo(HttpServletRequest request, Model model) throws Exception {
		String index = request.getParameter("index");

		System.out.println("받아온 공지사항 index: " + index);

		Statement st = null;
		ResultSet rs = null;
		Statement ddlst = null;
		int r;

		Connection dbconn = new DBconn.DBconnection().DBconn();

		try {

			ddlst = dbconn.createStatement();
			st = dbconn.createStatement();

			System.out.println("<<<<delete>>>>");

			// 공지사항 삭제 쿼리
			r = ddlst.executeUpdate("delete from refrigerator where num = " + Integer.valueOf(index) + ";");

			System.out.println("delete success!!");

			rs.close();
			st.close();
			ddlst.close();
			dbconn.close();
		} catch (Exception e) { // 예외가 발생하면 예외 상황을 처리한다.
			System.out.println("failed connect");
			e.printStackTrace();
		}
	}

}
